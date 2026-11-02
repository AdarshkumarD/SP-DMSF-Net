"""TensorFlow-version compatibility (needed for native-Windows GPU, TF <= 2.10).

TensorFlow 2.10 is the last release with native Windows GPU support. Its Keras
optimizers have no ``weight_decay`` argument (added in TF 2.11), so for TF < 2.11
we provide an Adam with decoupled weight decay (``var -= lr * wd * var`` applied
before the Adam step), which is the same rule that ``Adam(weight_decay=...)``
uses in TF >= 2.11.
"""
import warnings

import tensorflow as tf

TF_VERSION = tuple(int(x) for x in tf.__version__.split(".")[:2])
LEGACY_KERAS = TF_VERSION < (2, 11)


def make_decoupled_adam(base_adam):
    """Return an Adam subclass (of a Keras-2 style ``OptimizerV2`` Adam) with weight decay."""

    class DecoupledWeightDecayAdam(base_adam):
        def __init__(self, weight_decay=0.0, **kwargs):
            super().__init__(**kwargs)
            self._wd = float(weight_decay)

        def _resource_apply_dense(self, grad, var, apply_state=None):
            if self._wd:
                dev, dtype = var.device, var.dtype.base_dtype
                coef = ((apply_state or {}).get((dev, dtype))
                        or self._fallback_apply_state(dev, dtype))
                decay = var.assign_sub(coef["lr_t"] * self._wd * var,
                                       use_locking=self._use_locking)
                with tf.control_dependencies([decay]):
                    return super()._resource_apply_dense(grad, var, apply_state)
            return super()._resource_apply_dense(grad, var, apply_state)

        def get_config(self):
            cfg = super().get_config()
            cfg["weight_decay"] = self._wd
            return cfg

    return DecoupledWeightDecayAdam


def build_optimizer(tcfg):
    """Create the optimizer from the ``training`` config block (any TF version)."""
    name = tcfg["optimizer"].lower()
    lr = tcfg["learning_rate"]
    wd = tcfg.get("weight_decay", 0.0)
    mom = tcfg.get("momentum", 0.9)
    K = tf.keras.optimizers

    if LEGACY_KERAS:
        if name in ("adam", "adamw"):
            return make_decoupled_adam(K.Adam)(weight_decay=wd, learning_rate=lr)
        if name in ("sgd", "rmsprop") and wd:
            warnings.warn(f"weight_decay is ignored for {name} on TensorFlow < 2.11")
        if name == "sgd":
            return K.SGD(learning_rate=lr, momentum=mom)
        if name == "rmsprop":
            return K.RMSprop(learning_rate=lr, momentum=mom)
    else:
        if name == "adam":
            return K.Adam(learning_rate=lr, weight_decay=wd)
        if name == "adamw":
            return K.AdamW(learning_rate=lr, weight_decay=wd)
        if name == "sgd":
            return K.SGD(learning_rate=lr, momentum=mom, weight_decay=wd)
        if name == "rmsprop":
            return K.RMSprop(learning_rate=lr, momentum=mom)
    raise ValueError(f"Unknown optimizer: {name}")
