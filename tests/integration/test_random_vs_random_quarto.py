# """Tests d'intégration : partie complète RandomPlayer vs RandomPlayer sur QuartoEnv."""
# import pytest
# import numpy as np
# from torch.distributions import Categorical
# from deeprl_5iabd.envs.quarto import QuartoEnv
# from deeprl_5iabd.agents.random_agent import RandomPlayer


# @pytest.fixture
# def env():
#     return QuartoEnv()


# @pytest.fixture
# def player0():
#     return RandomPlayer(action_dim=32)


# @pytest.fixture
# def player1():
#     return RandomPlayer(action_dim=32)


# def play_full_game(env: QuartoEnv, player0: RandomPlayer, player1: RandomPlayer) -> int:
#     """Joue une partie complète et retourne le score final."""
#     env.reset()
#     max_steps = 200

#     for _ in range(max_steps):
#         if env.is_game_over():
#             break

#         current_player = player0 if env.player == 0 else player1
#         mask = env.get_action_space()
#         probs = current_player.forward(x=None, mask=mask)
#         dist = Categorical(probs)
#         action = dist.sample().item()
#         env.step(action)

#     return env.score()


# class TestRandomVsRandomQuarto:
#     def test_full_game_terminates(self, env, player0, player1):
#         """Une partie complète doit se terminer (is_game_over() = True)."""
#         env.reset()
#         steps = 0
#         max_steps = 200

#         while not env.is_game_over() and steps < max_steps:
#             current_player = player0 if env.player == 0 else player1
#             mask = env.get_action_space()
#             probs = current_player.forward(x=None, mask=mask)
#             dist = Categorical(probs)
#             action = dist.sample().item()
#             env.step(action)
#             steps += 1

#         assert env.is_game_over(), f"La partie n'était pas terminée après {max_steps} coups"

#     def test_score_is_valid(self, env, player0, player1):
#         """Le score final doit être dans {-1, 0, 1}."""
#         score = play_full_game(env, player0, player1)
#         assert score in {-1, 0, 1}, f"Score invalide : {score}"

#     def test_multiple_games_all_terminate(self, player0, player1):
#         """Plusieurs parties consécutives doivent toutes se terminer."""
#         for i in range(5):
#             env = QuartoEnv()
#             score = play_full_game(env, player0, player1)
#             assert score in {-1, 0, 1}, f"Partie {i+1} terminée avec score invalide : {score}"

#     def test_draw_has_no_available_pieces(self, env, player0, player1):
#         """En cas de match nul, toutes les pièces doivent être épuisées (available vide)."""
#         max_attempts = 30
#         draws_found = 0
#         for _ in range(max_attempts):
#             env.reset()
#             play_full_game(env, player0, player1)
#             if env.score() == 0:
#                 draws_found += 1
#                 assert np.all(env.available[:, :, 0] == -1), \
#                     "Match nul mais des pièces sont encore disponibles"
#         if draws_found == 0:
#             pytest.skip("Aucun match nul observé sur les 30 parties tentées")

#     def test_reinforce_env_compatible_with_random(self, env, player0, player1):
#         """Les espaces d'observation et d'action doivent être cohérents pendant la partie."""
#         env.reset()
#         steps = 0
#         max_steps = 200

#         while not env.is_game_over() and steps < max_steps:
#             obs = env.get_observation_space()
#             act = env.get_action_space()

#             assert len(obs) == 132, f"Taille observation incorrecte : {len(obs)}"
#             assert len(act) == 32, f"Taille action space incorrecte : {len(act)}"
#             assert sum(act) > 0, "Aucune action disponible alors que la partie continue"

#             current_player = player0 if env.player == 0 else player1
#             probs = current_player.forward(x=None, mask=act)
#             dist = Categorical(probs)
#             action = dist.sample().item()
#             env.step(action)
#             steps += 1
