#!/usr/bin/env python
"""Train and evaluate SP-DMSF-Net.

Examples
--------
    python train.py --config configs/rgbdem_4ch.yaml                 # RGB+DEM, aug as in config
    python train.py --config configs/rgbdem_4ch.yaml --aug off       # same, augmentation off
    python train.py --config configs/rgb_3ch.yaml                    # RGB only
    python train.py --config configs/rgb_3ch.yaml --set training.epochs=5 data.image_dir=/x/img
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from src.compat import build_optimizer
from src.data import prepare_datasets
from src.losses_metrics import make_loss, make_metrics
from src.model import build_from_config
from src.profiling import profile_hardware
from src.utils import (format_param_size, format_time, load_config, save_json,
                       seed_everything)
from src.visualize import plot_history, save_prediction_panel


def parse_args():
    p = argparse.ArgumentParser(description="Train SP-DMSF-Net")
    p.add_argument("--config", required=True, help="Path to a YAML config in configs/")
    p.add_argument("--aug", choices=["on", "off"], default=None,
                   help="Override augmentation.enabled from the config")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Override config values, e.g. training.epochs=5")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    if args.aug is not None:
        cfg.setdefault("augmentation", {})["enabled"] = (args.aug == "on")
    aug_on = bool(cfg.get("augmentation", {}).get("enabled", False))

    seed_everything(cfg["seed"])
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    run_name = f"{cfg['experiment_name']}_aug-{'on' if aug_on else 'off'}"
    out_dir = os.path.join(cfg["output_dir"], run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # ---- data -------------------------------------------------------------
    splits, ds, info, _ = prepare_datasets(cfg)
    save_json({k: [list(t) for t in v] for k, v in splits.items()},
              os.path.join(out_dir, "splits.json"))
    print(f"\nRun            : {run_name}")
    print(f"Samples        : {info['total']}  (train/val/test = "
          f"{info['train']}/{info['val']}/{info['test']})")
    print(f"Augmentation   : {info['augmentation_mode']}  "
          f"({info['train_samples_per_epoch']} train samples/epoch)")

    # ---- model ------------------------------------------------------------
    tcfg = cfg["training"]
    if tcfg.get("use_multi_gpu", False):
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()
    print(f"Devices        : {strategy.num_replicas_in_sync} replica(s)")

    with strategy.scope():
        model = build_from_config(cfg)
        model.compile(optimizer=build_optimizer(tcfg),
                      loss=make_loss(tcfg["loss"], strategy.num_replicas_in_sync),
                      metrics=make_metrics(tcfg.get("threshold", 0.5)))

    total = model.count_params()
    trainable = int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))
    print(f"Parameters     : {format_param_size(total)}  (trainable {format_param_size(trainable)})")

    hw = None
    hcfg = cfg.get("hardware_profiling", {})
    if hcfg.get("enabled", False):
        hw = profile_hardware(model, "SP_DMSF_NET", tcfg["batch_size"],
                              hcfg.get("warmup_runs", 20), hcfg.get("timed_runs", 100))

    # ---- train ------------------------------------------------------------
    weights_path = os.path.join(out_dir, "best.weights.h5")
    rl = tcfg["reduce_lr"]
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(weights_path, monitor="val_loss", save_best_only=True,
                                           save_weights_only=True, verbose=0),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                         patience=tcfg["early_stopping_patience"], verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=rl["factor"],
                                             patience=rl["patience"], min_lr=rl["min_lr"], verbose=1),
        tf.keras.callbacks.CSVLogger(os.path.join(out_dir, "history.csv")),
    ]
    t0 = time.time()
    history = model.fit(ds["train"], validation_data=ds["val"], epochs=tcfg["epochs"],
                        callbacks=callbacks, verbose=1)
    train_secs = time.time() - t0
    epochs_run = len(history.history["loss"])

    # ---- evaluate (best-val-loss weights) ---------------------------------
    model.load_weights(weights_path)
    val_res = model.evaluate(ds["val"], return_dict=True, verbose=0)
    test_res = model.evaluate(ds["test"], return_dict=True, verbose=0)
    train_last = {k: v[-1] for k, v in history.history.items() if not k.startswith("val_")}

    results = {
        "run": run_name, "input_channels": cfg["data"]["in_channels"],
        "augmentation": info["augmentation_mode"], "data": info,
        "epochs_run": epochs_run, "train_time": format_time(train_secs),
        "train_last_epoch": train_last, "validation": val_res, "test": test_res,
    }
    if hw is not None:
        hw["Total train time"] = format_time(train_secs)
        hw["Epochs run"] = epochs_run
        save_json(hw, os.path.join(out_dir, "hardware.json"))
    save_json(results, os.path.join(out_dir, "metrics.json"))

    print("\n" + "=" * 75)
    print(f"{run_name}  |  {results['train_time']}  |  {epochs_run} epochs")
    print("-" * 75)
    print(f"{'Metric':<18}{'Validation':>14}{'Test':>14}")
    for k in ["loss", "f1_score_metric", "soft_dice_score", "iou_score", "miou_metric",
              "pixel_accuracy", "precision_metric", "recall_metric"]:
        print(f"{k:<18}{val_res[k]:>14.4f}{test_res[k]:>14.4f}")
    print("=" * 75)

    # ---- figures ----------------------------------------------------------
    plot_history(history.history, os.path.join(out_dir, "training_curves.png"), run_name)
    save_prediction_panel(model, ds["test"], os.path.join(out_dir, "test_predictions.png"),
                          threshold=tcfg.get("threshold", 0.5), title=run_name)
    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
