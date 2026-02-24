import os
import platform
import xuance
import tensorflow as tf


def set_device(expected_device: str):
    """
    Set the computing device for a given deep learning framework.

    Args:
        dl_toolbox (str): The deep learning framework to use.
            Options: "torch", "tensorflow", "mindspore".
        expected_device (str): The desired computing device.
            Options: "cuda", "GPU", "gpu", "Ascend", "cpu", "CPU.

    Returns:
        str: The assigned computing device, which may differ from `expected_device`
        if the requested device is unavailable.
    """
    device = expected_device
    os.environ["TF_USE_LEGACY_KERAS"] = "1"  # Configure TensorFlow to use the legacy Keras 2 for tf.keras imports.
    if expected_device.upper() == "GPU":
        if len(tf.config.list_physical_devices('GPU')) == 0:
            device = "CPU"
            print("WARNING: GPU for Tensorflow2 is not available, set the device as 'CPU'.")
    elif expected_device.upper() == "CPU":
        device = "CPU"
    else:
        device = "CPU"
        print(f"WARNING: the device name {expected_device} is invalid, set the device as 'CPU'.")
    return device


def collect_device_info(
        rank: int = 0,
        agent=None,
) -> dict:
    """Collect runtime device / system info for reproducibility (TensorFlow 2.x).

    Returns a JSON-serializable dict.
    """
    info = {
        "Platform": platform.platform(),
        "Python": platform.python_version(),
        "XuanCe": xuance.__version__,
        "PID": os.getpid(),
        "Rank": rank,
    }

    try:
        info["TensorFlow"] = getattr(tf, "__version__", "unknown")

        # Physical devices visible to TF
        gpus = tf.config.list_physical_devices("GPU")
        cpus = tf.config.list_physical_devices("CPU")

        info["CUDA_Available"] = bool(gpus)
        info["num_gpus"] = len(gpus)
        info["num_cpus"] = len(cpus)

        # GPU details (best-effort; names are not always available)
        gpu_details = []
        for i, d in enumerate(gpus):
            # d.name often like '/physical_device:GPU:0'
            gpu_details.append({"index": i, "name": getattr(d, "name", str(d)), "device_type": "GPU"})
        info["gpus"] = gpu_details

        # Logical devices (useful when virtual GPUs / memory limits are set)
        logical_gpus = tf.config.list_logical_devices("GPU")
        info["num_logical_gpus"] = len(logical_gpus)

        # Build info sometimes contains cuda/cudnn versions (not always present)
        build_info = {}
        try:
            build_info = tf.sysconfig.get_build_info() or {}
        except Exception:
            build_info = {}

        # These keys vary across TF versions; keep it best-effort & JSON-safe
        if build_info:
            info["tf_build_info"] = {k: str(v) for k, v in build_info.items()}

        # Optional: record current visible devices env var (helps debug)
        info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", None)

        # Optional: if your Agent exposes its own device/strategy info, store it
        if agent is not None:
            # common patterns: agent.device / agent.strategy
            if hasattr(agent, "device"):
                info["agent_device"] = str(getattr(agent, "device"))
            if hasattr(agent, "strategy"):
                try:
                    info["tf_strategy"] = type(getattr(agent, "strategy")).__name__
                except Exception:
                    pass

    except Exception as e:
        # Keep it minimal but valid if TF isn't available or anything fails.
        info["CUDA_Available"] = False
        info["device_info_error"] = repr(e)

    return info
