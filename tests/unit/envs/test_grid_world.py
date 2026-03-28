"""Tests unitaires pour deeprl_5iabd.envs.quarto.QuartoEnv."""
import pytest
import numpy as np
from deeprl_5iabd.envs.grid_world import GridWorld

@pytest.fixture
def env():
    return GridWorld()

class TestGridWorldReset:

    def test_agent_position_at_start(self, env):
        """La position de l'agent doit être (0,0) au départ."""
        assert env.agent_pos == (0, 0)

    def test_board_at_start(self, env):
        """Le plateau doit avoir l'agent à la position (0,0) au départ."""
        assert env.board[(0,0)] == 1
        env.board[(0,0)] = -1
        assert np.all(env.board == -1)

    def test_reset_clears_board(self, env):
        """reset() reset le plateau."""
        env.step(0)
        env.reset()
        env.board[(0,0)] = -1
        assert np.all(env.board == -1)

    def test_reste_agent_position(self, env):
        """reset() reset la agent_pose a (0,0)"""
        env.step(0)
        env.reset()
        assert env.agent_pos == (0,0)

class TestGridWorldActionSpace:
    def test_action_space_at_start(self, env):
        """au début, l'agent est en (0,0), donc il ne peut pas aller en haut ou à gauche"""
        actions = env.get_action_space()
        assert actions == [1.0, 0.0, 1.0, 0.0]

    def test_action_space_row_0(self, env):
        """quand l'agent est sur la ligne 0, il ne peut pas aller a gauche"""
        for i in range(env.BOARD_SIZE - 1):
            actions = env.get_action_space()
            assert actions[3] == 0
            env.step(0)

    def test_action_space_col_0(self, env):
        """quand l'agent est sur la colonne 0, il ne peut pas aller en haut"""
        for i in range(env.BOARD_SIZE - 1):
            actions = env.get_action_space()
            assert actions[1] == 0
            env.step(2)

    def test_action_space_row_4(self, env):
        """quand l'agent est sur la ligne 4, il ne peut pas aller en bas"""
        for i in range(env.BOARD_SIZE - 1):
            env.step(0)

        for i in range(env.BOARD_SIZE - 1):
            actions = env.get_action_space()
            assert actions[0] == 0
            env.step(2)

    def test_action_space_col_4(self, env):
        """quand l'agent est sur la colonne 4, il ne peut pas aller a droite"""
        for i in range(env.BOARD_SIZE - 1):
            env.step(2)

        for i in range(env.BOARD_SIZE - 1):
            actions = env.get_action_space()
            assert actions[2] == 0
            env.step(1)

    def test_action_space_corner_0_4(self, env):
        """quand l'agent est en (0,4), il ne peut aller que dans 2 directions"""
        for i in range(env.BOARD_SIZE - 1):
            env.step(0)

        actions = env.get_action_space()
        assert actions == [0.0, 1.0, 1.0, 0.0]

    def test_action_space_all_allowed(self, env):
        """quand l'agent est au centre, il peut aller dans toutes les directions"""

        for r in range(1, env.BOARD_SIZE - 1):
            env.step(2)
            for _ in range(r):
                env.step(0)

            for c in range(1, env.BOARD_SIZE - 1):
                actions = env.get_action_space()
                assert actions == [1.0, 1.0, 1.0, 1.0]
                env.step(2)
            env.reset()

class TestGridWordStep:
    def test_step_down(self, env):
        """quand l'agent va en bas, il change de position et l'ancienne position est vide"""
        env.step(0)
        assert env.agent_pos == (1, 0)
        assert env.board[(1,0)] == 1
        env.board[(1,0)] = -1
        assert np.all(env.board == -1)

    def test_step_down_then_up(self, env):
        """quand l'agent va en haut, il change de position et l'ancienne position est vide"""
        env.step(0) #  déplace l'agent en bas
        assert env.agent_pos == (1, 0)
        assert env.board[(1,0)] == 1
        env.board[(1,0)] = -1
        assert np.all(env.board == -1)

        env.step(1) #  déplace l'agent en haut
        assert env.agent_pos == (0, 0)
        assert env.board[(0,0)] == 1
        env.board[(0,0)] = -1
        assert env.board[(1,0)] == -1

    def test_step_right(self, env):
        """quand l'agent va a droite, il change de position et l'ancienne position est vide"""
        env.step(2)
        assert env.agent_pos == (0, 1)
        assert env.board[(0,1)] == 1
        env.board[(0,1)] = -1
        assert np.all(env.board == -1)

    def test_step_right_then_left(self, env):
        """Sélectionner une pièce doit la retirer des pièces disponibles."""
        env.step(2) #  déplace l'agent à droite
        assert env.agent_pos == (0, 1)
        assert env.board[(0,1)] == 1
        env.board[(0,1)] = -1
        assert np.all(env.board == -1)

        env.step(3) #  déplace l'agent à gauche
        assert env.agent_pos == (0, 0)
        assert env.board[(0,0)] == 1
        env.board[(0,0)] = -1
        assert env.board[(1,0)] == -1

class TestGridWorldObservationSpace:
    def test_observation_space_length(self, env):
        """L'observation doit avoir 25 valeurs."""
        obs = env.get_observation_space()
        assert len(obs) == 2

    def test_observation_values_at_start(self, env):
        obs = env.get_observation_space()
        assert obs[0] == 0
        assert obs[1] == 0

    def test_observation_is_list(self, env):
        obs = env.get_observation_space()
        assert isinstance(obs, list)

    def test_observation_step_down(self, env):
        """quand l'agent va en bas, il change de position et l'ancienne position est vide"""
        env.step(0)
        obs = env.get_observation_space()
        assert obs[0] == 1
        assert obs[1] == 0

    def test_observation_step_down_then_up(self, env):
        """quand l'agent va en bas puis en haut, il change de position et l'ancienne position est vide"""
        env.step(0)
        obs = env.get_observation_space()
        assert obs[0] == 1
        assert obs[1] == 0

        env.step(1)
        obs = env.get_observation_space()
        assert obs[0] == 0
        assert obs[1] == 0

    def test_observation_step_right(self, env):
        """quand l'agent va à droit, il change de position et l'ancienne position est vide"""
        env.step(2)
        obs = env.get_observation_space()
        assert obs[0] == 0
        assert obs[1] == 1

    def test_observation_step_right_then_left(self, env):
        """quand l'agent va à droit puis à gauche, il change de position et l'ancienne position est vide"""
        
        env.step(2)
        obs = env.get_observation_space()
        assert obs[0] == 0
        assert obs[1] == 1

        env.step(3)
        obs = env.get_observation_space()
        assert obs[0] == 0
        assert obs[1] == 0

class TestGridWorldGameOver:
    def test_not_game_over_at_start(self, env):
        """Le jeu n'est pas terminé tans qu'on est pas en haut a droit ou en bas a droit"""
        T = [(0,4), (4,4)]
        for r in range(env.BOARD_SIZE):
            for _ in range(r):
                env.step(0)
            for c in range(env.BOARD_SIZE):
                if (r,c) not in T:
                    assert env.is_game_over() == False
                env.step(2)
            env.reset()

    def test_game_over_0_4(self, env):
        """en haut à droit game over"""
        for _ in range(env.BOARD_SIZE - 1):
            env.step(2)
        assert env.is_game_over()

    def test_game_over_4_4(self, env):
        """en bas à droit game over"""
        for _ in range(env.BOARD_SIZE - 1):
            env.step(0)
            env.step(2)
        assert env.is_game_over()

class TestGridWorldScore:
    def test_score_zero_at_start(self, env):
        """Le score doit être 0 tant que la partie n'est pas terminée."""
        T = [(0,4), (4,4)]
        for r in range(env.BOARD_SIZE):
            for _ in range(r):
                env.step(0)
            for c in range(env.BOARD_SIZE):
                if (r,c) not in T:
                    assert env.score() == 0
                env.step(2)
            env.reset()

    def test_score_win5_at_0_4(self, env):
        """Le score doit être 5 quand l'agent est en (0,4)."""
        for _ in range(env.BOARD_SIZE - 1):
            env.step(2)
        assert env.score() == 1

    def test_score_lose_minus_3_at_4_4(self, env):
        """Le score doit être -3 quand l'agent est en (4,4)."""
        for _ in range(env.BOARD_SIZE - 1):
            env.step(0)
            env.step(2)
        assert env.score() == -3