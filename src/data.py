"""Dataset discovery, splitting, preprocessing and tf.data pipeline."""
import os
import warnings
from glob import glob

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .augmentation import Augmenter, NO_AUGMENTATION, RANDOM_VARIANT

AUTOTUNE = tf.data.AUTOTUNE


# ---------------------------------------------------------------------------
# Path discovery & splitting
# ---------------------------------------------------------------------------
def _stem(p):
    return os.path.splitext(os.path.basename(p))[0]


def collect_samples(data_cfg):
    """Return a list of ``(image_path, mask_path, dem_path)`` tuples.

    ``dem_path`` is ``""`` for 3-channel (RGB-only) data. Files are paired by
    file stem; for RGB-only data, if stems do not match but both folders have
    the same number of files, the sorted order is used (with a warning).
    """
    imgs = sorted(glob(os.path.join(data_cfg["image_dir"], "*")))
    masks = sorted(glob(os.path.join(data_cfg["mask_dir"], "*")))
    if not imgs:
        raise FileNotFoundError(f"No images found in {data_cfg['image_dir']}")
    if not masks:
        raise FileNotFoundError(f"No masks found in {data_cfg['mask_dir']}")

    has_dem = bool(data_cfg.get("dem_dir"))
    mask_lut = {_stem(p): p for p in masks}
    dem_lut = {}
    if has_dem:
        dems = sorted(glob(os.path.join(data_cfg["dem_dir"], "*")))
        if not dems:
            raise FileNotFoundError(f"No DEM files found in {data_cfg['dem_dir']}")
        dem_lut = {_stem(p): p for p in dems}

    items = []
    for ip in imgs:
        s = _stem(ip)
        if s in mask_lut and (not has_dem or s in dem_lut):
            items.append((ip, mask_lut[s], dem_lut.get(s, "")))

    if not has_dem and len(items) != len(imgs):
        if len(imgs) == len(masks):
            warnings.warn("Image/mask file names do not match; pairing by sorted order.")
            items = [(i, m, "") for i, m in zip(imgs, masks)]
        else:
            raise RuntimeError(f"{len(imgs)} images but {len(masks)} masks and names do not match.")
    if not items:
        raise RuntimeError("No (image, mask, dem) triplets could be matched by file name.")
    return items


def make_splits(items, split_cfg, seed):
    """Two-stage split: ``holdout`` is carved out first, then divided into val/test.

    ``holdout`` and ``test_in_holdout`` are passed to scikit-learn as
    ``test_size`` (int = absolute count, float = fraction).
    """
    train, hold = train_test_split(items, test_size=split_cfg["holdout"],
                                   random_state=seed, shuffle=True)
    val, test = train_test_split(hold, test_size=split_cfg["test_in_holdout"],
                                 random_state=seed, shuffle=True)
    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# Sample loading / preprocessing
# ---------------------------------------------------------------------------
def _read_raw(path):
    low = path.lower()
    if low.endswith(".npy"):
        return np.load(path), False
    if low.endswith((".h5", ".hdf5")):
        import h5py
        with h5py.File(path, "r") as f:
            return np.array(f[list(f.keys())[0]]), False
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise IOError(f"Could not read file: {path}")
    return arr, True


def _read_rgb(path, h, w, hist_equalize):
    img, from_cv2 = _read_raw(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    elif img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))          # CHW -> HWC
    if img.shape[-1] == 4:
        img = img[:, :, :3]                          # drop alpha
    # OpenCV loads colour images as BGR. GeoTIFFs are left untouched (as in the
    # original experiments); every other format is converted to RGB.
    if from_cv2 and not path.lower().endswith((".tif", ".tiff")):
        img = np.ascontiguousarray(img[..., ::-1])
    img = cv2.resize(img.astype(np.float32), (w, h))

    # per-image, per-channel min-max normalisation to [0, 1]
    mn = img.min((0, 1), keepdims=True)
    mx = img.max((0, 1), keepdims=True)
    rng = mx - mn
    rng[rng == 0] = 1.0
    img = ((img - mn) / rng).astype(np.float32)

    if hist_equalize:
        for c in range(3):
            u8 = (img[:, :, c] * 255).astype(np.uint8)
            img[:, :, c] = cv2.equalizeHist(u8).astype(np.float32) / 255.0
    return img


def _read_dem(path, h, w):
    dem, _ = _read_raw(path)
    if dem.ndim == 3:
        dem = dem.mean(-1) if dem.shape[-1] > 1 else dem[:, :, 0]
    dem = cv2.resize(dem.astype(np.float32), (w, h))
    lo, hi = dem.min(), dem.max()
    if hi - lo > 0:
        dem = (dem - lo) / (hi - lo)
    return dem[:, :, np.newaxis].astype(np.float32)


def _read_mask(path, h, w):
    low = path.lower()
    if low.endswith((".npy", ".h5", ".hdf5")):
        mask, _ = _read_raw(path)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = np.transpose(mask, (1, 2, 0))
        if mask.ndim == 3:
            mask = mask[..., 0]
    else:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Could not read mask: {path}")
    mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask[:, :, np.newaxis] > 0).astype(np.float32)


def load_sample(img_p, mask_p, dem_p, data_cfg):
    """Return ``img (H,W,3)``, ``dem (H,W,1) or None``, ``mask (H,W,1)``."""
    h, w = data_cfg["img_height"], data_cfg["img_width"]
    hist_eq = bool(data_cfg.get("preprocessing", {}).get("hist_equalize", False))
    img = _read_rgb(img_p, h, w, hist_eq)
    dem = _read_dem(dem_p, h, w) if dem_p else None
    mask = _read_mask(mask_p, h, w)
    return img, dem, mask


# ---------------------------------------------------------------------------
# tf.data pipeline
# ---------------------------------------------------------------------------
def _to_str(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def make_parse_fn(data_cfg, augmenter):
    h, w, c = data_cfg["img_height"], data_cfg["img_width"], data_cfg["in_channels"]

    def parse_fn(img_p, mask_p, dem_p, aug_id):
        def _read(i, m, d, a):
            img, dem, mask = load_sample(_to_str(i), _to_str(m), _to_str(d), data_cfg)
            a = int(a)
            if a != NO_AUGMENTATION:
                img, dem, mask = augmenter.apply(img, dem, mask, a, np.random.default_rng())
            x = img if dem is None else np.concatenate([img, dem], -1)
            return x.astype(np.float32), mask.astype(np.float32)

        x, m = tf.numpy_function(_read, [img_p, mask_p, dem_p, aug_id],
                                 [tf.float32, tf.float32])
        x.set_shape([h, w, c])
        m.set_shape([h, w, 1])
        return x, m

    return parse_fn


def build_dataset(items, cfg, augmenter, training):
    ip = np.array([t[0] for t in items])
    mp = np.array([t[1] for t in items])
    dp = np.array([t[2] for t in items])

    if training and augmenter.enabled:
        if augmenter.mode == "offline_full":
            n = augmenter.n_variants
            ids = np.tile(np.arange(n), len(ip))
            ip, mp, dp = (np.repeat(x, n) for x in (ip, mp, dp))
        else:  # online_random
            ids = np.full(len(ip), RANDOM_VARIANT, dtype=np.int64)
    else:
        ids = np.full(len(ip), NO_AUGMENTATION, dtype=np.int64)

    ds = tf.data.Dataset.from_tensor_slices((ip, mp, dp, ids.astype(np.int32)))
    if training:
        ds = ds.shuffle(min(len(ip), 8192), seed=cfg["seed"], reshuffle_each_iteration=True)
    ds = ds.map(make_parse_fn(cfg["data"], augmenter), num_parallel_calls=AUTOTUNE)
    return ds.batch(cfg["training"]["batch_size"]).prefetch(AUTOTUNE)


def prepare_datasets(cfg):
    """Collect files, split, and build train/val/test ``tf.data`` datasets."""
    data_cfg = cfg["data"]
    square = data_cfg["img_height"] == data_cfg["img_width"]
    augmenter = Augmenter(cfg.get("augmentation"), square=square)

    items = collect_samples(data_cfg)
    splits = make_splits(items, data_cfg["split"], cfg["seed"])
    datasets = {
        "train": build_dataset(splits["train"], cfg, augmenter, training=True),
        "val": build_dataset(splits["val"], cfg, augmenter, training=False),
        "test": build_dataset(splits["test"], cfg, augmenter, training=False),
    }
    info = {
        "total": len(items),
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        "train_samples_per_epoch": augmenter.samples_per_epoch(len(splits["train"])),
        "augmentation_enabled": augmenter.enabled,
        "augmentation_mode": augmenter.mode if augmenter.enabled else "none",
        "n_variants": augmenter.n_variants if augmenter.enabled else 1,
    }
    return splits, datasets, info, augmenter
