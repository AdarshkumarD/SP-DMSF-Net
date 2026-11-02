"""Augmentation for RGB(+DEM) landslide segmentation.

Adapted from the 56-variant scheme of "UAV T-YOLO-Rice" (Sangaiah et al.,
IEEE TNSE 2024) for semantic segmentation:

* 4 rotations (0/90/180/270) x 2 mirror states  = 8 geometric bases
* each base -> none / gaussian / saturation / uniform / blur / bright_up /
  bright_down = 7 photometric versions               => 56 variants

Rules that keep the labels valid:

* geometric ops are applied identically to RGB, DEM and mask;
* photometric ops touch the RGB channels only (DEM and mask are never changed);
* augmentation is applied to the training split only, after the split.

The module only needs NumPy and OpenCV (no TensorFlow), so it can be tested on
its own. Whether it is used at all is controlled by ``augmentation.enabled``
in the YAML config (or ``--aug on|off`` on the command line).
"""
import cv2
import numpy as np

NO_AUGMENTATION = -2   # sentinel: leave the sample untouched (val / test / aug off)
RANDOM_VARIANT = -1    # sentinel: draw one random variant each time it is loaded

PHOTOMETRIC_OPS = ("none", "gaussian", "saturation", "uniform",
                   "blur", "bright_up", "bright_down")

DEFAULT_PARAMS = {
    "gauss_sigma": 0.05,       # std-dev on the [0, 1] scale
    "uniform_range": 0.10,     # noise ~ U(-r, +r)
    "saturation_factor": 1.5,
    "blur_ksize": 5,
    "bright_up": 1.3,
    "bright_down": 0.7,
}


def apply_geometric(arrays, k, flip):
    """Same rotation/mirror on every array (``None`` entries are passed through).

    Flip-then-rotate yields all 8 distinct orientations.
    """
    out = []
    for a in arrays:
        if a is None:
            out.append(None)
            continue
        if flip:
            a = np.flip(a, axis=1)
        out.append(np.ascontiguousarray(np.rot90(a, k)))
    return out


def apply_photometric(rgb, op, rng, params):
    """Photometric change on an RGB float32 image in [0, 1]."""
    if op == "none":
        return rgb
    if op == "gaussian":
        rgb = rgb + rng.normal(0.0, params["gauss_sigma"], rgb.shape)
    elif op == "uniform":
        r = params["uniform_range"]
        rgb = rgb + rng.uniform(-r, r, rgb.shape)
    elif op == "saturation":
        hsv = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * params["saturation_factor"], 0.0, 1.0)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    elif op == "blur":
        k = int(params["blur_ksize"])
        rgb = cv2.GaussianBlur(rgb.astype(np.float32), (k, k), 0)
    elif op == "bright_up":
        rgb = rgb * params["bright_up"]
    elif op == "bright_down":
        rgb = rgb * params["bright_down"]
    else:
        raise ValueError(f"Unknown photometric op: {op}")
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


class Augmenter:
    """Config-driven augmentation.

    Config keys (all optional except ``enabled``)::

        enabled: true | false
        mode: online_random | offline_full
        rotations: [0, 90, 180, 270]
        mirror: true
        photometric: [none, gaussian, saturation, uniform, blur, bright_up, bright_down]
        params: {gauss_sigma: 0.05, ...}

    ``online_random``: each epoch every training image gets one random variant
    (epoch size unchanged). ``offline_full``: every training image is expanded
    into all variants (epoch size x ``n_variants``; reduce ``training.epochs``).
    """

    def __init__(self, cfg=None, square=True):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.mode = cfg.get("mode", "online_random")
        if self.mode not in ("online_random", "offline_full"):
            raise ValueError("augmentation.mode must be online_random or offline_full")

        rotations = cfg.get("rotations", [0, 90, 180, 270])
        for r in rotations:
            if r not in (0, 90, 180, 270):
                raise ValueError(f"Invalid rotation {r}; use 0/90/180/270")
            if r in (90, 270) and not square:
                raise ValueError("90/270-degree rotations need square inputs "
                                 "(data.img_height == data.img_width)")
        photometric = list(cfg.get("photometric", PHOTOMETRIC_OPS))
        for p in photometric:
            if p not in PHOTOMETRIC_OPS:
                raise ValueError(f"Unknown photometric op {p!r}; options: {PHOTOMETRIC_OPS}")

        flips = (0, 1) if cfg.get("mirror", True) else (0,)
        self.geom_ops = [(int(r // 90), f) for f in flips for r in rotations]
        self.photo_ops = photometric
        self.variants = [(g, p) for g in self.geom_ops for p in self.photo_ops]
        self.n_variants = len(self.variants)
        self.params = {**DEFAULT_PARAMS, **(cfg.get("params") or {})}

    def samples_per_epoch(self, n_train):
        if self.enabled and self.mode == "offline_full":
            return n_train * self.n_variants
        return n_train

    def apply(self, img, dem, mask, variant_id, rng):
        """Return augmented ``(img, dem, mask)``. ``dem`` may be ``None`` (3-channel data)."""
        if variant_id == RANDOM_VARIANT:
            variant_id = int(rng.integers(self.n_variants))
        (k, flip), photo = self.variants[int(variant_id)]
        if k or flip:
            img, dem, mask = apply_geometric([img, dem, mask], k, flip)
        # Photometric ops run after min-max / histogram equalisation: otherwise
        # equalisation would almost completely cancel brightness changes.
        img = apply_photometric(img, photo, rng, self.params)
        return img, dem, mask
