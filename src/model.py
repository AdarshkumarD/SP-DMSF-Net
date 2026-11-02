"""SP-DMSF-Net: VGG16 encoder + ASPP bottleneck + DMSF decoder with parallel convs.

Two contributions relative to a standard U-Net-style decoder:

1. **DMSF block** (custom up-sampling, see :func:`dmsf_upsampling`): the
   encoder-path feature map is up-sampled by zero-insertion (1x1 transposed conv,
   stride 2), and the gaps are *filled with the projected skip features* by a
   fixed even/odd position mask (:class:`SelectiveMerge`), instead of leaving
   zeros or interpolating.
2. **Parallel convolution block** (:func:`parallel_conv_block`): a 1x1 and a 3x3
   convolution run side by side on the same input, are concatenated and fused by a
   3x3 convolution, replacing the usual series (3x3 -> 3x3) convolutions.
"""
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import register_keras_serializable


# ---------------------------------------------------------------------------
# Custom layers (serialisable, so saved models/weights load without Lambdas)
# ---------------------------------------------------------------------------
@register_keras_serializable(package="SPDMSF")
class SelectiveMerge(layers.Layer):
    """Keep ``up`` at (even, even) pixels, take ``skip`` everywhere else.

    ``up`` is the zero-inserted (2x up-sampled) tensor whose non-zero values sit on
    even rows AND even columns; the three remaining positions of every 2x2 cell
    are filled from ``skip``.
    """

    def call(self, inputs):
        up, skip = inputs
        shape = tf.shape(up)
        h, w = shape[1], shape[2]
        dt = up.dtype
        row_even = 1.0 - tf.math.mod(tf.cast(tf.range(h), dt), 2.0)
        col_even = 1.0 - tf.math.mod(tf.cast(tf.range(w), dt), 2.0)
        enc_mask = tf.reshape(tf.reshape(row_even, [h, 1]) * tf.reshape(col_even, [1, w]),
                              [1, h, w, 1])
        return up * enc_mask + skip * (1.0 - enc_mask)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


@register_keras_serializable(package="SPDMSF")
class ResizeLike(layers.Layer):
    """Bilinearly resize ``source`` to the spatial size of ``target``."""

    def call(self, inputs):
        source, target = inputs
        return tf.image.resize(source, tf.shape(target)[1:3], method="bilinear")

    def compute_output_shape(self, input_shape):
        s, t = input_shape
        return (s[0], t[1], t[2], s[3])


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def _conv_bn_relu(x, filters, kernel, name, dilation=1, use_bias=True):
    x = layers.Conv2D(filters, kernel, padding="same", dilation_rate=dilation,
                      use_bias=use_bias, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)


def parallel_conv_block(x, filters, name):
    """1x1 and 3x3 convs in parallel -> concat -> 3x3 fusion conv."""
    b1 = _conv_bn_relu(x, filters, 1, f"{name}_p1x1")
    b2 = _conv_bn_relu(x, filters, 3, f"{name}_p3x3")
    c = layers.Concatenate(name=f"{name}_cat")([b1, b2])
    return _conv_bn_relu(c, filters, 3, f"{name}_fuse")


def aspp_block(x, filters, name="aspp"):
    """ASPP with dilation rates 6/12/18 and global image pooling."""
    b0 = _conv_bn_relu(x, filters, 1, f"{name}_b0", use_bias=False)
    b1 = _conv_bn_relu(x, filters, 3, f"{name}_b1", dilation=6, use_bias=False)
    b2 = _conv_bn_relu(x, filters, 3, f"{name}_b2", dilation=12, use_bias=False)
    b3 = _conv_bn_relu(x, filters, 3, f"{name}_b3", dilation=18, use_bias=False)
    b4 = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name}_gap")(x)
    b4 = _conv_bn_relu(b4, filters, 1, f"{name}_b4", use_bias=False)
    b4 = ResizeLike(name=f"{name}_b4_resize")([b4, x])
    c = layers.Concatenate(name=f"{name}_cat")([b0, b1, b2, b3, b4])
    return _conv_bn_relu(c, filters, 1, f"{name}_project", use_bias=False)


def dmsf_upsampling(x, skip, filters, name):
    """DMSF block: zero-insertion up-sampling whose gaps are filled by the skip."""
    xm = layers.Conv2D(filters, 1, padding="same", use_bias=False, name=f"{name}_x_conv")(x)
    xm = layers.BatchNormalization(name=f"{name}_x_bn")(xm)
    xg = layers.Conv2DTranspose(filters, 1, strides=2, padding="same", use_bias=False,
                                name=f"{name}_x_up")(xm)
    sm = layers.Conv2D(filters, 1, padding="same", use_bias=False, name=f"{name}_skip_conv")(skip)
    sm = layers.BatchNormalization(name=f"{name}_skip_bn")(sm)
    return SelectiveMerge(name=f"{name}_merge")([xg, sm])


def decoder_block(x, skip, filters, name):
    up = dmsf_upsampling(x, skip, filters, name)
    cat = layers.Concatenate(name=f"{name}_skipcat")([skip, up])
    return parallel_conv_block(cat, filters, name)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
VGG_TAPS = ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3"]


def build_sp_dmsf_net(input_shape=(256, 256, 4), pretrained=True, aspp_filters=128,
                      dropout=0.3, name="SP_DMSF_NET"):
    """Build SP-DMSF-Net.

    ``input_shape`` channel count may be 3 (RGB) or 4 (RGB+DEM); 4-channel input
    is mapped to 3 channels by a learned 1x1 convolution before the VGG16 encoder.
    """
    h, w, c = input_shape
    inp = layers.Input(input_shape, name="input")
    x = inp if c == 3 else layers.Conv2D(3, 1, padding="same", activation="relu",
                                         name="channel_adapter")(inp)

    vgg = VGG16(weights="imagenet" if pretrained else None, include_top=False,
                input_shape=(h, w, 3))
    encoder = Model(vgg.input, [vgg.get_layer(n).output for n in VGG_TAPS],
                    name="vgg16_encoder")
    s1, s2, s3, s4, bottleneck = encoder(x)

    bottleneck = layers.Dropout(dropout, name="bottleneck_dropout")(bottleneck)
    a = aspp_block(bottleneck, aspp_filters)
    a = layers.Dropout(dropout, name="aspp_dropout")(a)

    d1 = decoder_block(a, s4, 512, "dec1")
    d2 = decoder_block(d1, s3, 256, "dec2")
    d3 = decoder_block(d2, s2, 128, "dec3")
    d4 = decoder_block(d3, s1, 64, "dec4")

    out = layers.Conv2D(1, 1, activation="sigmoid", name="mask")(d4)
    return Model(inp, out, name=name)


def build_from_config(cfg):
    d, m = cfg["data"], cfg["model"]
    return build_sp_dmsf_net(
        input_shape=(d["img_height"], d["img_width"], d["in_channels"]),
        pretrained=m.get("pretrained_backbone", True),
        aspp_filters=m.get("aspp_filters", 128),
        dropout=m.get("dropout", 0.3),
    )
