"""Tests unitaires pour deeprl_5iabd.envs.quarto.QuartoEnv."""
import numpy as np
import pytest
from deeprl_5iabd.envs.tictactoe import TicTacToe


@pytest.fixture
def env():
    return TicTacToe()

class TestTicTacToeReset:
    def test_board_empty_at_start(self, env):
        """Le plateau doit être initialement rempli de -1."""
        assert np.all(env.board == -1)

    def test_player_is_zero_at_start(self, env):
        """Le premier joueur doit être le joueur 0."""
        assert env.player == 0

    def test_reset_clears_board(self, env):
        """reset() doit vider le plateau."""
        # Simuler une sélection puis un placement
        env.step(0)
        env.reset()
        assert np.all(env.board == -1)

class TestTicTacToeActionSpace:
    def test_action_space_length(self, env):
        """get_action_space() doit retourner une liste de 32 valeurs."""
        assert len(env.get_action_space()) == 9
    
    def test_action_space_decrements_after_pick(self, env):
        """Après une sélection, une pièce doit disparaître de l'action space."""
        env.step(0)
        space = env.get_action_space()
        assert sum(space) == 8

class TestTicTacToeStep:
    def test_step_removes_from_available(self, env):
        """Sélectionner une pièce doit la retirer des pièces disponibles."""
        env.step(0)
        env.step(1)
        env.step(2)
        env.step(3)
        env.step(4)
        env.step(5)
        env.step(6)
        env.step(7)
        env.step(8)
        assert np.all(env.board != -1)

    def test_step_select_switches_player(self, env):
        """Sélectionner une pièce doit changer de joueur."""
        env.step(0)
        assert env.player == 1

    def test_step_place_puts_piece_on_board(self, env):
        """Placer une pièce doit la mettre sur le plateau."""
        env.step(0)
        assert np.all(env.board[0, 0] != -1)

class TestTicTacToeObservationSpace:
    def test_observation_space_length(self, env):
        """L'observation doit avoir 9 valeurs."""
        obs = env.get_observation_space()
        assert len(obs) == 9

    def test_observation_is_list(self, env):
        """get_observation_space() doit retourner une liste Python."""
        obs = env.get_observation_space()
        assert isinstance(obs, list)

class TestTicTacToeGameOver:
    def test_not_game_over_at_start(self, env):
        """Le jeu ne doit pas être terminé au départ."""
        assert not env.is_game_over()

    def test_score_zero_at_start(self, env):
        """Le score doit être 0 tant que la partie n'est pas terminée."""
        assert env.score() == 0

    def test_game_over_player0_win(self, env):
        # o o o
        # x x -
        # - - -
        env.step(0) # 0,0
        env.step(3)
        env.step(1) # 0,1
        env.step(4)
        env.step(2) # 0,2
        assert env.score() == 1

    def test_game_over_player1_win(self, env):
        # o o -
        # x x x
        # - o -
        env.step(0)
        env.step(3) # 0,1
        env.step(1)
        env.step(4) # 1,1
        env.step(7)
        env.step(5) # 2,1
        assert env.score() == -1

    def test_game_over_draw(self, env):
        # o o x
        # x o o
        # o x x
        env.step(0)
        env.step(3)
        env.step(1)
        env.step(2)
        env.step(4)
        env.step(8)
        env.step(6)
        env.step(7)
        assert env.score() == 0
