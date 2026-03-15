"""Tests unitaires pour deeprl_5iabd.envs.line_world.LineWorld."""
import pytest
from deeprl_5iabd.envs.line_world import LineWorld


@pytest.fixture
def env():
    return LineWorld()


class TestLineWorldReset:
    def test_initial_position(self, env):
        """La position initiale doit être 2 (centre)."""
        assert env.agent_pos == 2

    def test_reset_step_1_reset(self, env):
        """reset() doit ramener l'agent en position 2."""
        env.step(1)  # se déplacer
        env.reset()
        assert env.agent_pos == 2

    def test_reset_game_over_reset(self, env):
        """Après reset, le jeu ne doit pas être terminé."""
        env.agent_pos = 0  # état terminal
        env.reset()
        assert not env.is_game_over()


class TestLineWorldStep:
    def test_step_left(self, env):
        """L'action 0 (gauche) doit décrémenter la position."""
        env.step(0)
        assert env.agent_pos == 1

    def test_step_right(self, env):
        """L'action 1 (droite) doit incrémenter la position."""
        env.step(1)
        assert env.agent_pos == 3

    def test_step_ignored_when_game_over(self, env):
        """step() ne doit rien faire si la partie est terminée."""
        env.agent_pos = 0  # état terminal
        env.step(1)
        assert env.agent_pos == 0  # inchangé


class TestLineWorldGameOver:
    def test_game_over_at_position_0(self, env):
        """Position 0 = état terminal (défaite)."""
        env.agent_pos = 0
        assert env.is_game_over()

    def test_game_over_at_position_4(self, env):
        """Position 4 = état terminal (victoire)."""
        env.agent_pos = 4
        assert env.is_game_over()

    def test_not_game_over_in_middle(self, env):
        """Les positions 1, 2, 3 ne sont pas terminales."""
        for pos in [1, 2, 3]:
            env.agent_pos = pos
            assert not env.is_game_over(), f"Position {pos} ne devrait pas être terminale"


class TestLineWorldScore:
    def test_score_win(self, env):
        """Position 4 donne un score de +1."""
        env.agent_pos = 4
        assert env.score() == 1

    def test_score_loss(self, env):
        """Position 0 donne un score de -1."""
        env.agent_pos = 0
        assert env.score() == -1

    def test_score_neutral(self, env):
        """Positions 1-3 donnent un score de 0."""
        for pos in [1, 2, 3]:
            env.agent_pos = pos
            assert env.score() == 0


class TestLineWorldSpaces:
    def test_observation_space_is_position(self, env):
        """get_observation_space() doit retourner [agent_pos]."""
        assert env.get_observation_space() == [2]

    def test_observation_updates_after_step(self, env):
        """L'observation doit refléter la nouvelle position."""
        env.step(1)
        assert env.get_observation_space() == [3]

    def test_action_space_size(self, env):
        """get_action_space() doit toujours retourner une liste de taille 2."""
        assert len(env.get_action_space()) == 2

    def test_action_space_both_valid_in_center(self, env):
        """En position centrale, les deux actions doivent être disponibles."""
        assert env.get_action_space() == [1, 1]

    def test_action_space_at_left_boundary(self, env):
        """En position 1, l'action gauche vers 0 doit être disponible."""
        env.agent_pos = 1
        space = env.get_action_space()
        assert space[0] == 1  # peut aller à gauche (vers 0)
        assert space[1] == 1  # peut aller à droite

    def test_action_space_at_right_boundary(self, env):
        """En position 3, l'action droite vers 4 doit être disponible."""
        env.agent_pos = 3
        space = env.get_action_space()
        assert space[0] == 1
        assert space[1] == 1
