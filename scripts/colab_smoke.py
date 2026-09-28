import platform
import torch

print({"python": platform.python_version(), "cuda": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
