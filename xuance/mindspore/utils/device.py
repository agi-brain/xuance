import os
import platform
import xuance
import mindspore as ms


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

    if expected_device.upper() == "GPU":
        try:
            ms.set_context(device_target="GPU")
        except:
            device = "CPU"
            print("WARNING: GPU for MindSpore is not available, set the device as 'CPU'.")
    elif expected_device.upper() == "ASCEND":
        try:
            ms.set_context(device_target="Ascend")
        except:
            device = "CPU"
            print("WARNING: Ascend for MindSpore is not available, set the device as 'CPU'.")
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
    """Collect runtime device / system info for reproducibility (MindSpore).

    Returns a JSON-serializable dict.
    """
    info = {
        "Platform": platform.platform(),
        "Python": platform.python_version(),
        "XuanCe": xuance.__version__,
        "MindSpore": getattr(ms, "__version__", "unknown"),
        "PID": os.getpid(),
        "Rank": rank,

        # Helpful for debugging device visibility in different launchers/CI.
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", None),
        "ascend_visible_devices": os.environ.get("ASCEND_VISIBLE_DEVICES", None),
        "device_id_env": os.environ.get("DEVICE_ID", None),
        "rank_id_env": os.environ.get("RANK_ID", None),
        "rank_size_env": os.environ.get("RANK_SIZE", None),
    }

    try:
        # MindSpore runtime device context (source of truth).
        device_target = ms.context.get_context("device_target")  # e.g., "CPU", "GPU", "Ascend"
        device_id = ms.context.get_context("device_id")  # int or None

        info["device_target"] = str(device_target)
        info["device_id"] = int(device_id) if device_id is not None else None

        # Simple availability flags derived from device_target.
        # (MindSpore does not provide a universal cuda.is_available() like PyTorch.)
        tgt = str(device_target).upper()
        info["GPU_Available"] = (tgt == "GPU")
        info["Ascend_Available"] = (tgt == "ASCEND")

    except Exception as e:
        # Keep it minimal but valid if context querying fails.
        info["device_info_error"] = repr(e)

    # Best-effort: record configured device in agent/config if your framework defines it.
    try:
        if agent is not None:
            cfg = getattr(agent, "config", None)
            if cfg is not None and hasattr(cfg, "device"):
                info["configured_device"] = str(getattr(cfg, "device"))
    except Exception:
        pass

    return info
