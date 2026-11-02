"""Unit tests for src/augmentation.py (NumPy + OpenCV only).

Run:  python tests/test_augmentation.py     (or: pytest tests)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.augmentation import Augmenter, NO_AUGMENTATION, RANDOM_VARIANT  # noqa: E402


def _sample(seed=0, size=32):
    r = np.random.default_rng(seed)
    img = r.random((size, size, 3)).astype(np.float32)
    dem = r.random((size, size, 1)).astype(np.float32)
    mask = (r.random((size, size, 1)) > 0.7).astype(np.float32)
    return img, dem, mask


def test_variant_count():
    assert Augmenter({"enabled": True}).n_variants == 56
    assert Augmenter({"enabled": True, "mirror": False}).n_variants == 28
    assert Augmenter({"enabled": True, "photometric": ["none"]}).n_variants == 8


def test_geometric_keeps_alignment_and_dem_untouched_photometrically():
    aug = Augmenter({"enabled": True})
    rng = np.random.default_rng(0)
    img, dem, mask = _sample()
    for vid in range(aug.n_variants):
        (k, flip), photo = aug.variants[vid]
        i2, d2, m2 = aug.apply(img, dem, mask, vid, rng)
        assert i2.shape == img.shape and d2.shape == dem.shape and m2.shape == mask.shape
        # DEM and mask undergo the same geometric transform, no photometric change
        ref = dem
        ref_m = mask
        if flip:
            ref, ref_m = np.flip(ref, 1), np.flip(ref_m, 1)
        assert np.allclose(d2, np.rot90(ref, k))
        assert np.allclose(m2, np.rot90(ref_m, k))
        assert set(np.unique(m2)) <= {0.0, 1.0}
        assert i2.min() >= 0.0 and i2.max() <= 1.0


def test_identity_variant():
    aug = Augmenter({"enabled": True})
    img, dem, mask = _sample()
    i2, d2, m2 = aug.apply(img, dem, mask, 0, np.random.default_rng(0))
    assert np.allclose(i2, img) and np.allclose(d2, dem) and np.allclose(m2, mask)


def test_random_variant_and_no_dem():
    aug = Augmenter({"enabled": True})
    img, _, mask = _sample()
    i2, d2, m2 = aug.apply(img, None, mask, RANDOM_VARIANT, np.random.default_rng(1))
    assert d2 is None and i2.shape == img.shape and m2.shape == mask.shape


def test_samples_per_epoch():
    assert Augmenter({"enabled": False}).samples_per_epoch(100) == 100
    assert Augmenter({"enabled": True, "mode": "online_random"}).samples_per_epoch(100) == 100
    assert Augmenter({"enabled": True, "mode": "offline_full"}).samples_per_epoch(100) == 5600
    assert NO_AUGMENTATION != RANDOM_VARIANT


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
