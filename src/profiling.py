"""Hardware-complexity profiling: params, size, FLOPs/MACs, latency, FPS, GPU memory."""
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import \
    convert_variables_to_constants_v2_as_graph


def get_model_flops(model, batch=1):
    """Total FLOPs of one forward pass (TF profiler). MACs ~= FLOPs / 2."""
    spec = tf.TensorSpec([batch] + list(model.inputs[0].shape[1:]), model.inputs[0].dtype)
    concrete = tf.function(lambda x: model(x, training=False)).get_concrete_function(spec)
    _, graph_def = convert_variables_to_constants_v2_as_graph(concrete)
    with tf.Graph().as_default() as g:
        tf.graph_util.import_graph_def(graph_def, name="")
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        prof = tf.compat.v1.profiler.profile(
            graph=g, run_meta=tf.compat.v1.RunMetadata(), cmd="scope", options=opts)
    return int(prof.total_float_ops)


def _bench(fn, x, n_warm, n_runs):
    for _ in range(n_warm):
        _ = fn(x).numpy()                       # warm-up (tracing, cuDNN autotune)
    ts = []
    for _ in range(n_runs):
        t = time.perf_counter()
        _ = fn(x).numpy()                       # .numpy() forces a GPU sync
        ts.append((time.perf_counter() - t) * 1000.0)
    return np.array(ts)


def profile_hardware(model, name, batch_size, warmup_runs=20, timed_runs=100):
    h, w, c = model.inputs[0].shape[1:]
    total = model.count_params()
    trainable = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
    non_train = int(np.sum([np.prod(v.shape) for v in model.non_trainable_weights]))

    try:
        flops = get_model_flops(model, batch=1)
    except Exception as e:  # profiler support varies across TF/Keras versions
        print(f"[warn] FLOPs profiling failed: {e}")
        flops = float("nan")

    fn = tf.function(lambda x: model(x, training=False))
    x1 = tf.random.uniform([1, h, w, c])
    xb = tf.random.uniform([batch_size, h, w, c])

    gpus = tf.config.list_physical_devices("GPU")
    peak_mb = float("nan")
    if gpus:
        try:
            tf.config.experimental.reset_memory_stats("GPU:0")
        except Exception:
            pass

    t1 = _bench(fn, x1, warmup_runs, timed_runs)
    tb = _bench(fn, xb, max(5, warmup_runs // 4), max(20, timed_runs // 4))

    if gpus:
        try:
            peak_mb = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1024 ** 2
        except Exception:
            pass
    try:
        dev = (tf.config.experimental.get_device_details(gpus[0]).get("device_name", "GPU")
               if gpus else "CPU")
    except Exception:
        dev = "GPU" if gpus else "CPU"

    hw = {
        "Model": name.upper(),
        "Device": dev,
        "Input": f"{h}x{w}x{c}",
        "Total Params": total,
        "Trainable Params": trainable,
        "Non-trainable Params": non_train,
        "Size FP32 (MB)": round(total * 4 / 1024 ** 2, 2),
        "Size FP16 (MB)": round(total * 2 / 1024 ** 2, 2),
        "Size INT8 est. (MB)": round(total * 1 / 1024 ** 2, 2),
        "FLOPs (G)": round(flops / 1e9, 3),
        "MACs (G)": round(flops / 2e9, 3),
        "Latency bs=1 mean (ms)": round(float(t1.mean()), 2),
        "Latency bs=1 std (ms)": round(float(t1.std()), 2),
        "Latency bs=1 p95 (ms)": round(float(np.percentile(t1, 95)), 2),
        "FPS bs=1": round(1000.0 / float(t1.mean()), 2),
        f"Throughput bs={batch_size} (img/s)": round(batch_size * 1000.0 / float(tb.mean()), 2),
        "Peak GPU mem during inference (MB)": round(peak_mb, 1),
    }
    print("\n" + "-" * 60)
    print(f"HARDWARE COMPLEXITY  |  {hw['Model']}  |  {hw['Device']}")
    print("-" * 60)
    for k, v in hw.items():
        if k not in ("Model", "Device"):
            print(f"{k:<38}: {v}")
    print("-" * 60 + "\n")
    return hw
