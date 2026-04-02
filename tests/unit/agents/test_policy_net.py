# """Tests unitaires pour deeprl_5iabd.agents.policy_net.PolicyNetwork."""
# import torch
# import pytest
# import tempfile
# import os
# from pathlib import Path
# from unittest.mock import patch
# from deeprl_5iabd.agents.policy_net import PolicyNetwork


# @pytest.fixture
# def net():
#     return PolicyNetwork(name="test", input_size=8, output_size=4, hiddenlayers=[16, 8])


# class TestPolicyNetworkForward:
#     def test_output_shape(self, net):
#         """La sortie doit avoir la taille output_size."""
#         x = torch.randn(8)
#         mask = [1, 1, 1, 1]
#         out = net.forward(x, mask)
#         assert out.shape == (4,)

#     def test_probs_sum_to_one(self, net):
#         """La somme des probabilités doit être ≈ 1."""
#         x = torch.randn(8)
#         mask = [1, 1, 0, 1]
#         out = net.forward(x, mask)
#         assert abs(out.sum().item() - 1.0) < 1e-5

#     def test_masked_positions_zero(self, net):
#         """Les positions masquées (0) doivent avoir une prob nulle."""
#         x = torch.randn(8)
#         mask = [1, 0, 1, 0]
#         out = net.forward(x, mask)
#         assert out[1].item() == pytest.approx(0.0, abs=1e-6)
#         assert out[3].item() == pytest.approx(0.0, abs=1e-6)

#     def test_non_masked_positions_positive(self, net):
#         """Les positions actives doivent avoir une prob strictement positive."""
#         x = torch.randn(8)
#         mask = [1, 0, 1, 0]
#         out = net.forward(x, mask)
#         assert out[0].item() > 0
#         assert out[2].item() > 0

#     def test_default_hidden_layers(self):
#         """Sans hiddenlayers, doit utiliser [512, 256, 128] par défaut."""
#         net = PolicyNetwork(name="default", input_size=4, output_size=2)
#         assert net.config["hiddenlayers"] == [512, 256, 128]


# class TestPolicyNetworkConfig:
#     def test_config_contains_name(self, net):
#         assert net.config["name"] == "test"

#     def test_config_contains_input_size(self, net):
#         assert net.config["input_size"] == 8

#     def test_config_contains_output_size(self, net):
#         assert net.config["output_size"] == 4

#     def test_config_contains_hiddenlayers(self, net):
#         assert net.config["hiddenlayers"] == [16, 8]


# class TestPolicyNetworkSaveLoad:
#     def test_save_and_load_config(self, net, tmp_path):
#         """Sauvegarder et recharger doit conserver la configuration."""
#         filename = "test_net.pth"
#         with patch("deeprl_5iabd.agents.policy_net.settings") as mock_settings:
#             mock_settings.models_path = tmp_path
#             net.save(filename)
#             loaded = PolicyNetwork.load(filename)

#         assert loaded.config["name"] == net.config["name"]
#         assert loaded.config["input_size"] == net.config["input_size"]
#         assert loaded.config["output_size"] == net.config["output_size"]

#     def test_save_and_load_weights(self, net, tmp_path):
#         """Sauvegarder et recharger doit conserver les poids du réseau."""
#         filename = "test_weights.pth"
#         with patch("deeprl_5iabd.agents.policy_net.settings") as mock_settings:
#             mock_settings.models_path = tmp_path
#             net.save(filename)
#             loaded = PolicyNetwork.load(filename)

#         x = torch.randn(8)
#         mask = [1, 1, 1, 1]
#         out_original = net.forward(x, mask)
#         out_loaded = loaded.forward(x, mask)
#         assert torch.allclose(out_original, out_loaded, atol=1e-5)
