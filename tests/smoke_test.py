"""End-to-end smoke test on synthetic data (no downloads, no real dataset needed).

Creates tiny fake 3-channel and 4-channel datasets, then runs train.py for 1-2
epochs with both augmentation settings, followed by evaluate.py.

    python tests/smoke_test.py
"""
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
N = 24
SIZE = 64


def make_dummy(root, with_dem):
    rng = np.random.default_rng(0)
    for sub in ["image", "mask"] + (["dem"] if with_dem else []):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    for i in range(N):
        img = (rng.random((SIZE, SIZE, 3)) * 255).astype(np.uint8)
        mask = np.zeros((SIZE, SIZE), np.uint8)
        x, y = rng.integers(5, 30, 2)
        mask[y:y + 20, x:x + 20] = 255
        cv2.imwrite(os.path.join(root, "image", f"{i:03d}.png"), img)
        cv2.imwrite(os.path.join(root, "mask", f"{i:03d}.png"), mask)
        if with_dem:
            dem = (rng.random((SIZE, SIZE)) * 1000).astype(np.float32)
            cv2.imwrite(os.path.join(root, "dem", f"{i:03d}.tif"), dem)


def run(cmd):
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        d3, d4, out = os.path.join(tmp, "rgb"), os.path.join(tmp, "rgbdem"), os.path.join(tmp, "out")
        make_dummy(d3, False)
        make_dummy(d4, True)
        common = ["training.epochs=2", "training.batch_size=4", "model.pretrained_backbone=false",
                  "hardware_profiling.enabled=false", f"output_dir={out}",
                  f"data.img_height={SIZE}", f"data.img_width={SIZE}",
                  "data.split.holdout=8", "data.split.test_in_holdout=4"]
        run([sys.executable, "train.py", "--config", "configs/rgbdem_4ch.yaml", "--aug", "on",
             "--set", f"data.image_dir={d4}/image", f"data.mask_dir={d4}/mask",
             f"data.dem_dir={d4}/dem", *common])
        run([sys.executable, "train.py", "--config", "configs/rgbdem_4ch.yaml", "--aug", "off",
             "--set", f"data.image_dir={d4}/image", f"data.mask_dir={d4}/mask",
             f"data.dem_dir={d4}/dem", *common])
        run([sys.executable, "train.py", "--config", "configs/rgb_3ch.yaml", "--aug", "on",
             "--set", f"data.image_dir={d3}/image", f"data.mask_dir={d3}/mask",
             "data.split.holdout=8", "data.split.test_in_holdout=4",
             *[c for c in common if not c.startswith("data.split")]])
        run([sys.executable, "evaluate.py", "--config", "configs/rgbdem_4ch.yaml",
             "--weights", os.path.join(out, "sp_dmsf_net_rgbdem_4ch_aug-on", "best.weights.h5"),
             "--set", f"data.image_dir={d4}/image", f"data.mask_dir={d4}/mask",
             f"data.dem_dir={d4}/dem", *[c for c in common if c.startswith("data.")]])
        print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
