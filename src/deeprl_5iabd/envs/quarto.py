import os
import pygame
import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
from deeprl_5iabd.helper import ImageButton
from deeprl_5iabd.config import settings
from enum import IntEnum
from deeprl_5iabd.helper import Player


class Phase(IntEnum):
    SELECT = 0
    PLACE  = 1


class QuartoEnv(gym.Env):
    BOARD_SIZE    = 4
    NUM_PIECES    = 16
    NUM_ATTRS     = 4

    PG_PIECE_W    = 120
    PG_PIECE_H    = 168
    PG_GAP        = 4
    PG_WINDOW_W   = (PG_PIECE_W + PG_GAP) * (BOARD_SIZE * 2) + PG_PIECE_W
    PG_WINDOW_H   = (PG_PIECE_H + PG_GAP) * BOARD_SIZE + PG_PIECE_H

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.screen = None
        self._offscreen = None
        self.render_mode = render_mode

        if self.render_mode == "human":
            self._init_pygame()
        elif self.render_mode == "rgb_array":
            self._init_offscreen()

        self.phase = Phase.SELECT
        self.current_piece = np.zeros(5, dtype=np.float32)
        self.board = np.zeros(16*5, dtype=np.float32)
        self.pieces: np.ndarray = np.array([
            1,1,1,1,1 ,1,1,1,0,1 ,1,1,0,1,1 ,1,1,0,0,1,
            1,0,1,1,1 ,1,0,1,0,1 ,1,0,0,1,1 ,1,0,0,0,1,
            0,1,1,1,1 ,0,1,1,0,1 ,0,1,0,1,1 ,0,1,0,0,1,
            0,0,1,1,1 ,0,0,1,0,1 ,0,0,0,1,1 ,0,0,0,0,1
        ])

        self.action_space = spaces.Discrete(16, dtype=np.int32)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(1 + len(self.current_piece) + len(self.board) + len(self.pieces),),
            dtype=np.float32
        )

        self._obs_buffer = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        self._action_mask_buffer = np.zeros(self.action_space.n, dtype=np.int8)
        self._win_lines = [
            (0,1,2,3), (4,5,6,7), (8,9,10,11), (12,13,14,15),
            (0,4,8,12), (1,5,9,13), (2,6,10,14), (3,7,11,15),
            (0,5,10,15), (3,6,9,12)
        ]

        self.current_player = Player.PLAYER_1
        self.agent_player = Player.PLAYER_1
        self.is_multi_player = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.pieces[4::5] = 1
        self.board[:] = 0
        self.current_piece[:] = 0

        self.phase = Phase.SELECT
        self.p_counter = 16
        return self._get_obs(), {}

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        if self.phase == Phase.PLACE:
            self.board[action*5:action*5+5] = self.current_piece
            self.current_piece[:] = 0

            is_win = self._check_win()
            if is_win and self.current_player == self.agent_player:
                reward = 1.0
                terminated = True
                info["msg"] = f"Agent a gagné !"
            elif is_win and self.current_player != self.agent_player:
                reward = -1.0
                terminated = True
                info["msg"] = f"Agent a perdu !"
            elif self.p_counter == 0:
                terminated = True
                info["msg"] = "Match Nul !"
            else:
                self.phase = Phase.SELECT

        else:
            self.p_counter -= 1
            self.current_piece[:] = self.pieces[action*5:action*5+5]
            self.pieces[action*5+4] = 0

            self.phase = Phase.PLACE
            self.current_player = Player.PLAYER_2 if self.current_player == Player.PLAYER_1 else Player.PLAYER_1

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode not in ("human", "rgb_array"):
            return

        surface = self.screen if self.render_mode == "human" else self._offscreen
        surface.fill((0, 0, 0))

        font = pygame.font.SysFont(None, 36)
        phase_label = {
            Phase.SELECT: "choisissez une pièce",
            Phase.PLACE:  "placez la pièce",
        }.get(self.phase, "Fin de partie")

        surface.blit(font.render(f"Joueur {self.current_player.value} — {phase_label}", True, (255, 255, 255)), (10, 10))

        for r in range(4):
            for c in range(4):
                self.pg_board[r][c].image  = self._asset(self.board[(r*4+c)*5 : (r*4+c)*5+5])
                self.pg_pieces[r][c].image = self._asset(self.pieces[(r*4+c)*5 : (r*4+c)*5+5])
                self.pg_board[r][c].draw(surface)
                self.pg_pieces[r][c].draw(surface)

        self.pg_selected.image = self._asset(self.current_piece[:5])
        self.pg_selected.draw(surface)

        if self.render_mode == "human":
            pygame.display.flip()
        else:
            return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))

    def close(self):
        if self.screen is not None or self._offscreen is not None:
            pygame.quit()
            self.screen = None
            self._offscreen = None


    def get_action_mask(self):
        if self.phase == Phase.SELECT:
            self._action_mask_buffer[:] = self.pieces[4::5]
        else:
            self._action_mask_buffer[:] = (self.board[4::5] == 0).astype(np.int8)
        return self._action_mask_buffer

    def _get_obs(self):
        np.concatenate(([self.phase.value], self.current_piece, self.board, self.pieces), out=self._obs_buffer)
        return self._obs_buffer

    def state_id(self) -> int:
        return hash(self._obs_buffer.tobytes())

    def _check_win(self):
        for (i1, i2, i3, i4) in self._win_lines:
            if self.board[i1*5 + 4] == 0:   continue
            if self.board[i2*5 + 4] == 0:   continue
            if self.board[i3*5 + 4] == 0:   continue
            if self.board[i4*5 + 4] == 0:   continue

            for i in range(4):
                if self.board[i1*5+i] == self.board[i2*5+i] == self.board[i3*5+i] == self.board[i4*5+i]:
                    return True

        return False

    def _asset(self, piece: np.ndarray):
        if piece[4] == 0:
            return None
        return self.pg_assets.get("".join(map(str, piece[:4].astype(int))))

    def _init_assets(self):
        self.pg_assets = {
            f"{i:04b}": pygame.transform.scale(
                pygame.image.load(f"{settings.quarto_assets_path}/{i:04b}.png"),
                (self.PG_PIECE_W, self.PG_PIECE_H)
            )
            for i in range(self.NUM_PIECES)
        }
        self.pg_assets["-1-1-1-1"] = None

        self.pg_board = [[ImageButton(c * (self.PG_PIECE_W + self.PG_GAP),
                                       (r + 1) * (self.PG_PIECE_H + self.PG_GAP),
                                       self.PG_PIECE_W, self.PG_PIECE_H)
                          for c in range(4)] for r in range(4)]

        self.pg_pieces = [[ImageButton((c + 5) * (self.PG_PIECE_W + self.PG_GAP),
                                        (r + 1) * (self.PG_PIECE_H + self.PG_GAP),
                                        self.PG_PIECE_W, self.PG_PIECE_H)
                           for c in range(4)] for r in range(4)]

        self.pg_selected = ImageButton(self.PG_WINDOW_W - self.PG_PIECE_W, 0,
                                        self.PG_PIECE_W, self.PG_PIECE_H)

    def _init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.PG_WINDOW_W, self.PG_WINDOW_H))
        pygame.display.set_caption("Quarto")
        self._init_assets()

    def _init_offscreen(self):
        pygame.init()
        self._offscreen = pygame.Surface((self.PG_WINDOW_W, self.PG_WINDOW_H))
        self._init_assets()

    def _wait_for_human_click(self, mask: np.ndarray) -> int:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    raise SystemExit
                for r in range(4):
                    for c in range(4):
                        if self.phase == Phase.SELECT:
                            if self.pg_pieces[r][c].is_clicked(event):
                                if mask[r * 4 + c] == 1:
                                    return r * 4 + c
                        else:
                            if self.pg_board[r][c].is_clicked(event):
                                if mask[r * 4 + c] == 1:
                                    return r * 4 + c

    def __str__(self):
        return "QuartoEnv"