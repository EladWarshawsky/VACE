from typing import Union
import torch


def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        # Return a proper CUDA device, not an int index
        idx = torch.cuda.current_device()
        return torch.device(f"cuda:{idx}")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")