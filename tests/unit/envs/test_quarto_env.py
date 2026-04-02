"""Tests unitaires pour deeprl_5iabd.envs.quarto.QuartoEnv."""
import torch
import numpy as np
import pytest
from deeprl_5iabd.envs.quarto import QuartoEnv


@pytest.fixture
def env():
    return QuartoEnv()

class TestQuarto:
    def test_quarto_init(self, env):
        assert torch.all(env.selected == torch.full((4,), -1))
        assert torch.all(env.board == torch.full((16 * 4,), -1))
        assert torch.all(env.available_ps == env.ALL_PIECES)
        assert env._action_mask_buffer.shape == (16 + 16,)
        assert env._obs_mask_buffer.shape == (4 + 16 * 4 + 16 * 4,)

        assert env.selecting == True
        assert env.player in [0, 1]
        assert env._pygame_ready == False
    
    def test_quarto_reset(self, env):
        env.board[0] = 1
        env.selected[0] = 1
        env.available_ps[0] = 1

        old_board_ptr = env.board.data_ptr()
        old_selected_ptr = env.selected.data_ptr()
        old_available_ptr = env.available_ps.data_ptr()

        env.reset()
        assert torch.all(env.selected == torch.full((4,), -1))
        assert env.selected.data_ptr() == old_selected_ptr

        assert torch.all(env.board == torch.full((16 * 4,), -1))
        assert env.board.data_ptr() == old_board_ptr

        assert torch.all(env.available_ps == env.ALL_PIECES)
        assert env.available_ps.data_ptr() == old_available_ptr

    def test_quarto_determinize(self, env):
        env.board[0] = 1
        env.selected[0] = 1
        env.available_ps[0] = 1

        new_env = env.determinize()
        assert torch.all(new_env.board == env.board)
        assert torch.all(new_env.available_ps == env.available_ps)
        assert torch.all(new_env.selected == env.selected)
        assert new_env.player == env.player
        assert new_env.selecting == env.selecting
        assert new_env._pygame_ready == env._pygame_ready

    def test_quarto_step_select(self, env):
        current_player = env.current_player
        selected = env.selected
        selected_data_ptr = selected.data_ptr()
        available_ps_data_ptr = env.available_ps.data_ptr()
        board_data_ptr = env.board.data_ptr()

        env.selecting = True

        env.step(16)
        assert env.selecting == False
        assert env.current_player != current_player
        assert torch.all(env.selected == env.ALL_PIECES[0:4])
        assert env.selected.data_ptr() == selected_data_ptr
        assert torch.all(env.available_ps[0:4] == -1)
        assert env.available_ps.data_ptr() == available_ps_data_ptr
        assert torch.all(env.board == -1)
        assert env.board.data_ptr() == board_data_ptr

    def test_quarto_step_place(self, env):
        selected_data_ptr = env.selected.data_ptr()
        available_ps_data_ptr = env.available_ps.data_ptr()
        board_data_ptr = env.board.data_ptr()

        env.selecting = False
        env.selected = env.selected.copy_(env.ALL_PIECES[0:4])
        env.step(0)
    
        assert env.selecting == True
        assert torch.all(env.selected == -1)
        assert env.selected.data_ptr() == selected_data_ptr
        assert torch.all(env.board[0:4] == env.ALL_PIECES[0:4])
        assert env.board.data_ptr() == board_data_ptr
        assert env.available_ps.data_ptr() == available_ps_data_ptr

    def test_quarto_get_action_mask_select(self, env):
        # [board, available_ps]
        env.selecting = True
        aa_mask_ptr = env._action_mask_buffer.data_ptr()

        aa_mask = env.get_action_mask()
        assert torch.all(aa_mask[0:16] == 0)
        assert torch.all(aa_mask[-16:] == 1)
        assert aa_mask_ptr == env._action_mask_buffer.data_ptr()

    def test_quarto_get_action_mask_place(self, env):
        # [board, available_ps]
        env.selecting = False
        aa_mask_ptr = env._action_mask_buffer.data_ptr()

        aa_mask = env.get_action_mask()
        assert torch.all(aa_mask[0:16] == 1)
        assert torch.all(aa_mask[-16:] == 0)
        assert aa_mask_ptr == env._action_mask_buffer.data_ptr()

    def test_quarto_get_observation_space(self, env):
        obs_ptr = env._obs_mask_buffer.data_ptr()
        obs = env.get_observation_mask()
        assert torch.all(obs[0:4] == env.selected)
        assert torch.all(obs[4:4+16*4] == env.board.flatten())
        assert torch.all(obs[4+16*4:] == env.available_ps)
        assert obs_ptr == env._obs_mask_buffer.data_ptr()

    def test_quarto_is_game_over_win(self, env):
        env.step(16)
        env.step(0)
        env.step(17)
        env.step(1)
        env.step(18)
        env.step(2)
        env.step(19)
        env.step(3)
        assert env.is_game_over == True
