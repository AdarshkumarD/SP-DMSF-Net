#!/usr/bin/env python
"""Evaluate trained SP-DMSF-Net weights on the validation or test split.

The split is re-created from the config (same seed), so use the same config
that was used for training.

    python evaluate.py --config configs/rgbdem_4ch.yaml \
        --weights outputs/sp_dmsf_net_rgbdem_4ch_aug-on/best.weights.h5 --split test
"""
import argparse
import os

import tensorflow as tf

from src.data import prepare_datasets
from src.losses_metrics import make_loss, make_metrics
from src.model import build_from_config
from src.utils import load_config, save_json, seed_everything
from src.visualize import save_prediction_panel


def main():
    p = argparse.ArgumentParser(description="Evaluate SP-DMSF-Net")
    p.add_argument("--config", required=True)
    p.add_argument("--weights", required=True, help="Path to a .weights.h5 file from train.py")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--out-dir", default=None, help="Default: folder of the weights file")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    cfg["model"]["pretrained_backbone"] = False   # weights come from the file
    seed_everything(cfg["seed"])

    _, ds, info, _ = prepare_datasets(cfg)
    model = build_from_config(cfg)
    model.load_weights(args.weights)   # load first, then compile (avoids optimizer-state warning)
    model.compile(loss=make_loss(cfg["training"]["loss"]),
                  metrics=make_metrics(cfg["training"].get("threshold", 0.5)))

    res = model.evaluate(ds[args.split], return_dict=True, verbose=0)
    print(f"\n{args.split.upper()} results ({info[args.split]} samples):")
    for k, v in res.items():
        print(f"  {k:<18}{v:.4f}")

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.weights))
    os.makedirs(out_dir, exist_ok=True)
    save_json(res, os.path.join(out_dir, f"eval_{args.split}.json"))
    save_prediction_panel(model, ds[args.split], os.path.join(out_dir, f"eval_{args.split}_predictions.png"),
                          threshold=cfg["training"].get("threshold", 0.5))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
