"""Segmentation losses and metrics (binary landslide masks)."""
import tensorflow as tf

EPS = 1e-6


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def make_metrics(threshold=0.5):
    """Return the list of Keras metric functions (names are used as history keys)."""

    def _flat(y_true, y_pred, binarise):
        yt = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        yp = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        if binarise:
            yp = tf.cast(yp > threshold, tf.float32)
        return yt, yp

    def _confusion(y_true, y_pred):
        yt, yp = _flat(y_true, y_pred, True)
        tp = tf.reduce_sum(yt * yp)
        fp = tf.reduce_sum((1. - yt) * yp)
        fn = tf.reduce_sum(yt * (1. - yp))
        tn = tf.reduce_sum((1. - yt) * (1. - yp))
        return tp, fp, fn, tn

    def soft_dice_score(y_true, y_pred):
        yt, yp = _flat(y_true, y_pred, False)
        i = tf.reduce_sum(yt * yp)
        return (2. * i + EPS) / (tf.reduce_sum(yt) + tf.reduce_sum(yp) + EPS)

    def iou_score(y_true, y_pred):
        yt, yp = _flat(y_true, y_pred, False)
        i = tf.reduce_sum(yt * yp)
        return (i + EPS) / (tf.reduce_sum(yt) + tf.reduce_sum(yp) - i + EPS)

    def precision_metric(y_true, y_pred):
        tp, fp, _, _ = _confusion(y_true, y_pred)
        return (tp + EPS) / (tp + fp + EPS)

    def recall_metric(y_true, y_pred):
        tp, _, fn, _ = _confusion(y_true, y_pred)
        return (tp + EPS) / (tp + fn + EPS)

    def f1_score_metric(y_true, y_pred):
        p = precision_metric(y_true, y_pred)
        r = recall_metric(y_true, y_pred)
        return (2. * p * r + EPS) / (p + r + EPS)

    def miou_metric(y_true, y_pred):
        """Mean IoU over the two classes (landslide, background)."""
        tp, fp, fn, tn = _confusion(y_true, y_pred)
        return 0.5 * ((tp + EPS) / (tp + fp + fn + EPS) + (tn + EPS) / (tn + fn + fp + EPS))

    def pixel_accuracy(y_true, y_pred):
        tp, fp, fn, tn = _confusion(y_true, y_pred)
        return (tp + tn + EPS) / (tp + tn + fp + fn + EPS)

    return [soft_dice_score, iou_score, miou_metric, precision_metric,
            recall_metric, f1_score_metric, pixel_accuracy]


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def _soft_dice(y_true, y_pred):
    yt = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    yp = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    i = tf.reduce_sum(yt * yp)
    return (2. * i + EPS) / (tf.reduce_sum(yt) + tf.reduce_sum(yp) + EPS)


def make_loss(loss_cfg, num_replicas=1):
    """Build the training loss from the ``training.loss`` config block.

    ``num_replicas`` rescales the Dice term so that the loss value is comparable
    between single- and multi-GPU (MirroredStrategy) runs.
    """
    focal_alpha = loss_cfg.get("focal", {}).get("alpha", 0.85)
    focal_gamma = loss_cfg.get("focal", {}).get("gamma", 2.0)
    tv_alpha = loss_cfg.get("tversky", {}).get("alpha", 0.3)
    tv_beta = loss_cfg.get("tversky", {}).get("beta", 0.7)

    def bce_loss(y_true, y_pred):
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(
            tf.cast(y_true, tf.float32), y_pred))

    def dice_loss(y_true, y_pred):
        return (1.0 - _soft_dice(y_true, y_pred)) / num_replicas

    def focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
        bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        a_t = y_true * focal_alpha + (1.0 - y_true) * (1.0 - focal_alpha)
        return tf.reduce_mean(a_t * tf.pow(1.0 - p_t, focal_gamma) * bce)

    def tversky_loss(y_true, y_pred):
        yt = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        yp = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        tp = tf.reduce_sum(yt * yp)
        fp = tf.reduce_sum((1.0 - yt) * yp)
        fn = tf.reduce_sum(yt * (1.0 - yp))
        return 1.0 - ((tp + EPS) / (tp + tv_alpha * fp + tv_beta * fn + EPS))

    mode = loss_cfg["mode"].lower()
    if mode == "bce":
        return bce_loss
    if mode == "dice":
        return dice_loss
    if mode == "focal":
        return focal_loss
    if mode == "tversky":
        return tversky_loss

    # combined: weighted sum; a weight of 0 disables that term
    w = loss_cfg.get("combined", {})
    terms = [(w.get("bce", 0.0), bce_loss), (w.get("dice", 0.0), dice_loss),
             (w.get("focal", 0.0), focal_loss), (w.get("tversky", 0.0), tversky_loss)]

    def combined_loss(y_true, y_pred):
        total = 0.0
        for weight, fn in terms:
            if weight:
                total += weight * fn(y_true, y_pred)
        return total

    return combined_loss
