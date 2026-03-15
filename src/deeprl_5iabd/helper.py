import torch
import torch.nn.functional as F
import numpy as np

def get_default_device() -> str:
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        return torch.accelerator.current_accelerator().type
    return "cpu"

def softmax_with_mask(S, M):
    M = torch.tensor(M, dtype=S.dtype, device=S.device)
    positive_or_null_s = S - S.min()
    masked_positive_or_null_s = positive_or_null_s * M
    negative_or_null_s = masked_positive_or_null_s - masked_positive_or_null_s.max()
    exp_s = torch.exp(negative_or_null_s)
    masked_exp_s = exp_s * M
    return masked_exp_s / masked_exp_s.sum()