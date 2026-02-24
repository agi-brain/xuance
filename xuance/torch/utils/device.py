import os
import platform
import torch
import xuance


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
    if ("cuda" in expected_device) or (expected_device.upper() == "GPU"):
        if not torch.cuda.is_available():
            print("WARNING: CUDA for PyTorch is not available, set the device as 'cpu'.")
            device = "cpu"
        elif expected_device.upper() == "GPU":
            print(f"WARNING: the device name {expected_device} is invalid, set the device as 'cuda:0'.")
            device = "cuda:0"
    elif expected_device.upper() == "CPU":
        device = "cpu"
    else:
        print(f"WARNING: the device name {expected_device} is invalid, set the device as 'cpu'.")
        device = "cpu"
    return device


def collect_device_info(
        rank: int = 0,
        agent=None,
) -> dict:
    """Collect runtime device / system info for reproducibility.

    Returns a JSON-serializable dict.
    """
    info = {
        "Platform": platform.platform(),
        "CUDA_Available": bool(torch.cuda.is_available()),
        "Python": platform.python_version(),
        "XuanCe": xuance.__version__,
        "PyTorch": getattr(torch, "__version__", "unknown"),
        "PID": os.getpid(),
        "Rank": rank,
    }

    # Try to use agent's real device (most reliable).
    device = None
    try:
        # Find a parameter device if possible.
        if agent is not None:
            # Try common attribute names in your codebase
            obj = getattr(agent, "policy", None)
            if obj is not None and hasattr(obj, "parameters"):
                device = next(obj.parameters()).device

            # Fallback: config.device if no parameter found
            if device is None:
                device = torch.device(str(agent.config.device))

        if device is None:
            device = torch.device("cpu")

        info["device"] = str(device)

        if device.type == "cuda":
            idx = device.index if device.index is not None else 0
            info.update({
                "gpu_index": int(idx),
                "gpu_name": torch.cuda.get_device_name(idx),
                "cuda_version": getattr(torch.version, "cuda", None),
            })
            # Optional: driver / capability
            try:
                cap = torch.cuda.get_device_capability(idx)
                info["gpu_capability"] = f"{cap[0]}.{cap[1]}"
            except Exception:
                pass
        else:
            info.update({
                "cpu_arch": platform.processor(),
            })

    except Exception as e:
        # If torch isn't available or something fails, keep it minimal but valid.
        info["device"] = str(getattr(getattr(agent, 'config', None), "device", "unknown"))
        info["device_info_error"] = repr(e)

    return info
