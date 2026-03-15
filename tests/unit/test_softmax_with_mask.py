import torch
import numpy as np
import pytest
from deeprl_5iabd.helper import softmax_with_mask


def test_softmax_sums_to_one():
    """La somme des probabilités doit être exactement 1."""
    S = torch.tensor([1.0, 2.0, 3.0, 4.0])
    M = torch.tensor([1.0, 1.0, 0.0, 1.0])
    probs = softmax_with_mask(S, M)
    assert abs(probs.sum().item() - 1.0) < 1e-6


def test_masked_positions_are_zero():
    """Les positions masquées (M=0) doivent avoir une probabilité nulle."""
    S = torch.tensor([1.0, 2.0, 3.0, 4.0])
    M = torch.tensor([1.0, 1.0, 0.0, 1.0])
    probs = softmax_with_mask(S, M)
    assert probs[2].item() == 0.0


def test_probabilities_are_non_negative():
    """Toutes les probabilités doivent être >= 0."""
    S = torch.tensor([-5.0, 0.0, 3.0, 10.0])
    M = torch.tensor([1.0, 1.0, 1.0, 0.0])
    probs = softmax_with_mask(S, M)
    assert (probs >= 0).all()


def test_only_one_valid_action():
    """Avec un seul masque actif, la probabilité doit être 1.0."""
    S = torch.tensor([2.0, 5.0, 1.0, 3.0])
    M = torch.tensor([0.0, 1.0, 0.0, 0.0])
    probs = softmax_with_mask(S, M)
    assert abs(probs[1].item() - 1.0) < 1e-6
    assert abs(probs.sum().item() - 1.0) < 1e-6


def test_all_actions_valid_matches_softmax():
    """Sans masquage, doit être équivalent à softmax standard."""
    S = torch.tensor([1.0, 2.0, 3.0])
    M = torch.tensor([1.0, 1.0, 1.0])
    probs = softmax_with_mask(S, M)
    expected = torch.softmax(S, dim=0)
    assert torch.allclose(probs, expected, atol=1e-5)


def test_multiple_masked_positions():
    """Plusieurs positions masquées doivent toutes être nulles."""
    S = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    M = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
    probs = softmax_with_mask(S, M)
    assert probs[0].item() == 0.0
    assert probs[2].item() == 0.0
    assert probs[4].item() == 0.0
    assert abs(probs.sum().item() - 1.0) < 1e-6


def test_accepts_numpy_mask():
    """Le masque peut être fourni comme array numpy."""
    S = torch.tensor([1.0, 2.0, 3.0])
    M = np.array([1.0, 0.0, 1.0])
    probs = softmax_with_mask(S, M)
    assert probs[1].item() == 0.0
    assert abs(probs.sum().item() - 1.0) < 1e-6
