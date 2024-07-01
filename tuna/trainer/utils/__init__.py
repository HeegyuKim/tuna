import torch

from .logger import BaseLogger
from .checkpoint import upload_local_directory_to_gcs

def convert_dict_tensor_devices(d, device: str):
    for k, v in d.items():
        if torch.is_tensor(v) and v.device != device:
            d[k] = v.to(device)

    return d


def detach_tensors(d):
    for k, v in d.items():
        if torch.is_tensor(v):
            d[k] = v.detach().cpu()

    return d