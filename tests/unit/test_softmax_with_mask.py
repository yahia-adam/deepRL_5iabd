import torch
import numpy as np
from deeprl_5iabd.helper import softmax_with_mask

def test_softmax_sums_to_one():
    S = torch.tensor([1.0, 2.0, 3.0, 4.0])
    M = torch.tensor([1.0, 1.0, 0.0, 1.0])
    probs = softmax_with_mask(S, M)
    assert abs(probs.sum().item() - 1.0) < 1e-6

def test_masked_positions_are_zero():
    S = torch.tensor([1.0, 2.0, 3.0, 4.0])
    M = torch.tensor([1.0, 1.0, 0.0, 1.0])
    probs = softmax_with_mask(S, M)
    assert probs[2].item() == 0.0
