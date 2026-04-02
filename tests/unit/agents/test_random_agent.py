# """Tests unitaires pour deeprl_5iabd.agents.random_agent.RandomPlayer."""
# import torch
# import pytest
# from deeprl_5iabd.agents.random_agent import RandomPlayer


# @pytest.fixture
# def agent():
#     return RandomPlayer(action_dim=6)


# class TestRandomPlayerForward:
#     def test_output_shape(self, agent):
#         """La sortie doit avoir la taille action_dim."""
#         mask = [1, 1, 0, 1, 0, 1]
#         out = agent.forward(x=None, mask=mask)
#         assert out.shape == (6,)

#     def test_probs_sum_to_one(self, agent):
#         """La somme des probabilités doit être ≈ 1."""
#         mask = [1, 1, 0, 1, 0, 1]
#         out = agent.forward(x=None, mask=mask)
#         assert abs(out.sum().item() - 1.0) < 1e-5

#     def test_masked_positions_zero(self, agent):
#         """Les positions masquées (0) doivent avoir une prob nulle."""
#         mask = [1, 1, 0, 1, 0, 1]
#         out = agent.forward(x=None, mask=mask)
#         assert out[2].item() == pytest.approx(0.0, abs=1e-6)
#         assert out[4].item() == pytest.approx(0.0, abs=1e-6)

#     def test_active_positions_positive(self, agent):
#         """Les positions actives doivent avoir une prob > 0."""
#         mask = [0, 1, 0, 1, 0, 0]
#         out = agent.forward(x=None, mask=mask)
#         assert out[1].item() > 0
#         assert out[3].item() > 0

#     def test_ignores_x_argument(self, agent):
#         """L'argument x est ignoré (peut être None)."""
#         mask = [1, 0, 1, 0, 1, 0]
#         out1 = agent.forward(x=None, mask=mask)
#         x_tensor = torch.randn(6)
#         out2 = agent.forward(x=x_tensor, mask=mask)
#         # Les deux doivent avoir la même structure (masked zeros)
#         assert out1[1].item() == pytest.approx(0.0, abs=1e-6)
#         assert out2[1].item() == pytest.approx(0.0, abs=1e-6)

#     def test_all_actions_valid(self, agent):
#         """Sans masquage, toutes les actions ont une prob > 0."""
#         mask = [1, 1, 1, 1, 1, 1]
#         out = agent.forward(x=None, mask=mask)
#         assert (out > 0).all()

#     def test_single_valid_action(self, agent):
#         """Un seul masque actif → probabilité = 1.0."""
#         mask = [0, 0, 0, 1, 0, 0]
#         out = agent.forward(x=None, mask=mask)
#         assert abs(out[3].item() - 1.0) < 1e-5

#     def test_action_dim_stored(self):
#         """action_dim doit être correctement stocké."""
#         rp = RandomPlayer(action_dim=32)
#         assert rp.action_dim == 32
