"""Tests unitaires pour deeprl_5iabd.envs.quarto.QuartoEnv."""
import numpy as np
import pytest
from deeprl_5iabd.envs.quarto import QuartoEnv


@pytest.fixture
def env():
    return QuartoEnv()


class TestQuartoReset:
    def test_board_empty_at_start(self, env):
        """Le plateau doit être initialement rempli de -1."""
        assert np.all(env.board == -1)

    def test_available_full_at_start(self, env):
        """Les 16 pièces doivent être disponibles au départ."""
        # Aucune cellule dans available ne doit avoir first attr = -1
        count_available = np.sum(env.available[:, :, 0] != -1)
        assert count_available == 16

    def test_selecting_is_true_at_start(self, env):
        """Au démarrage, la phase est 'choisir une pièce'."""
        assert env.selecting is True

    def test_player_is_zero_at_start(self, env):
        """Le premier joueur doit être le joueur 0."""
        assert env.player == 0

    def test_selected_empty_at_start(self, env):
        """Aucune pièce ne doit être sélectionnée au départ."""
        assert np.all(env.selected == -1)

    def test_reset_clears_board(self, env):
        """reset() doit vider le plateau."""
        # Simuler une sélection puis un placement
        env.step(0)   # sélection pièce (0,0)
        env.step(16)  # placement case (0,0)
        env.reset()
        assert np.all(env.board == -1)

    def test_reset_restores_available(self, env):
        """reset() doit restaurer les 16 pièces disponibles."""
        env.step(0)  # retire une pièce
        env.reset()
        count_available = np.sum(env.available[:, :, 0] != -1)
        assert count_available == 16


class TestQuartoActionSpace:
    def test_action_space_length(self, env):
        """get_action_space() doit retourner une liste de 32 valeurs."""
        assert len(env.get_action_space()) == 32

    def test_action_space_selecting_phase(self, env):
        """En phase sélection, les 16 premières positions codent les pièces dispo."""
        space = env.get_action_space()
        picks = space[:16]
        placements = space[16:]
        assert sum(picks) == 16         # 16 pièces disponibles
        assert sum(placements) == 0     # aucun placement possible

    def test_action_space_placing_phase(self, env):
        """En phase placement, les 16 dernières positions codent les cases vides."""
        env.step(0)  # sélectionne pièce index 0
        space = env.get_action_space()
        picks = space[:16]
        placements = space[16:]
        assert sum(picks) == 0          # plus de sélection possible
        assert sum(placements) == 16    # 16 cases vides

    def test_action_space_decrements_after_pick(self, env):
        """Après une sélection, une pièce doit disparaître de l'action space."""
        env.step(0)   # sélectionne pièce (0,0)
        env.step(16)  # place en (0,0)
        # Maintenant en phase sélection avec 15 pièces
        space = env.get_action_space()
        assert sum(space[:16]) == 15


class TestQuartoStep:
    def test_step_select_removes_from_available(self, env):
        """Sélectionner une pièce doit la retirer des pièces disponibles."""
        piece_before = env.available[0, 0].copy()
        env.step(0)
        assert np.all(env.available[0, 0] == -1)
        assert np.all(env.selected == piece_before)

    def test_step_select_switches_player(self, env):
        """Sélectionner une pièce doit changer de joueur."""
        env.step(0)
        assert env.player == 1

    def test_step_place_puts_piece_on_board(self, env):
        """Placer une pièce doit la mettre sur le plateau."""
        env.step(0)   # sélectionne pièce (0,0)
        env.step(16)  # place en case (0,0) du plateau
        assert np.all(env.board[0, 0] != -1)

    def test_step_place_clears_selected(self, env):
        """Après un placement, selected doit être réinitialisé à -1."""
        env.step(0)
        env.step(16)
        assert np.all(env.selected == -1)


class TestQuartoObservationSpace:
    def test_observation_space_length(self, env):
        """L'observation doit avoir 132 valeurs (4 + 64 + 64)."""
        obs = env.get_observation_space()
        assert len(obs) == 4 + 64 + 64  # selected + board + available

    def test_observation_is_list(self, env):
        """get_observation_space() doit retourner une liste Python."""
        obs = env.get_observation_space()
        assert isinstance(obs, list)


class TestQuartoGameOver:
    def test_not_game_over_at_start(self, env):
        """Le jeu ne doit pas être terminé au départ."""
        assert not env.is_game_over()

    def test_score_zero_at_start(self, env):
        """Le score doit être 0 tant que la partie n'est pas terminée."""
        assert env.score() == 0

    def test_game_over_when_all_pieces_placed(self, env):
        """La partie s'arrête quand toutes les pièces sont placées (match nul)."""
        # Jouer toute une partie sans victoire
        # On place les 16 pièces sur le plateau
        for i in range(16):
            pick_idx = next(
                r * 4 + c
                for r in range(4)
                for c in range(4)
                if env.available[r, c, 0] != -1
            )
            env.step(pick_idx)
            place_idx = next(
                16 + r * 4 + c
                for r in range(4)
                for c in range(4)
                if env.board[r, c, 0] == -1
            )
            env.step(place_idx)
        assert env.is_game_over()
