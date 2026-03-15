"""Tests unitaires pour deeprl_5iabd.helper."""
import torch
from deeprl_5iabd.helper import get_default_device


def test_get_default_device_returns_string():
    """get_default_device doit retourner une chaîne non vide."""
    device = get_default_device()
    assert isinstance(device, str)
    assert len(device) > 0


def test_get_default_device_is_valid_torch_device():
    """Le device retourné doit être accepté par PyTorch."""
    device = get_default_device()
    # Ne doit pas lever d'exception
    tensor = torch.zeros(1, device=device)
    assert tensor.device.type == device.split(":")[0]


def test_get_default_device_known_values():
    """Le device retourné doit être parmi les valeurs connues."""
    device = get_default_device()
    known_prefixes = ("cpu", "cuda", "mps", "xpu", "npu")
    assert device.startswith(known_prefixes), f"Device inconnu : {device}"
