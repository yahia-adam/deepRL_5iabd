import time
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from deeprl_5iabd.helper import Player
from deeprl_5iabd.config import settings
from deeprl_5iabd.helper import ImageButton
from enum import IntEnum


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class GridWorldEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    BOARD_SIZE = 5

    PG_PIECE_W = 150
    PG_PIECE_H = 150
    PG_GAP = 5

    PG_WINDOW_W = (PG_PIECE_W + PG_GAP) * BOARD_SIZE
    PG_WINDOW_H = (PG_PIECE_H + PG_GAP) * BOARD_SIZE

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.screen = None
        self._offscreen = None
        self.last_action = Action.DOWN

        if self.render_mode == "human":
            self._init_pygame()
        elif self.render_mode == "rgb_array":
            self._init_offscreen()

        self.pos = 0

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(25,), dtype=np.float32)

        self._action_mask_buffer = np.ones(self.action_space.n, dtype=np.float32)

        self.board = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.board[0] = 1

        self.agent_player = Player.PLAYER_1
        self.current_player = Player.PLAYER_1
        self.is_multi_player = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = 0
        self.board[:] = 0
        self.board[self.pos] = 1
        return self._get_obs(), {}

    def step(self, action):
        self.board[self.pos] = 0

        if action == Action.UP.value and self.pos not in [0, 1, 2, 3, 4]:
            self.pos -= 5
        elif action == Action.DOWN.value and self.pos not in [20, 21, 22, 23, 24]:
            self.pos += 5
        elif action == Action.LEFT.value and self.pos not in [0, 5, 10, 15, 20]:
            self.pos -= 1
        elif action == Action.RIGHT.value and self.pos not in [4, 9, 14, 19, 24]:
            self.pos += 1

        self.board[self.pos] = 1

        terminated = False
        truncated = False
        reward = 0.0

        if self.pos == 24:
            terminated = True
            reward = 1.0
        elif self.pos == 4:
            terminated = True
            reward = -1.0

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode not in ("human", "rgb_array"):
            return

        surface = self.screen if self.render_mode == "human" else self._offscreen
        surface.fill((30, 30, 30))

        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if self.board[r * self.BOARD_SIZE + c] == 1:
                    self.pg_board[r][c].image = self.pg_assets[self.last_action.value]
                else:
                    self.pg_board[r][c].image = None
                self.pg_board[r][c].draw(surface)

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
        return self._action_mask_buffer

    def _get_obs(self):
        return self.board

    def state_id(self):
        return self.pos

    def _init_assets(self):
        self.pg_assets = [
            pygame.transform.scale(
                pygame.image.load(f"{settings.grid_world_assets_path}/{i}.png"),
                (self.PG_PIECE_W, self.PG_PIECE_H),
            )
            for i in range(0, 4)
        ]
        self.pg_board = [
            [
                ImageButton(
                    c * (self.PG_PIECE_W + self.PG_GAP),
                    r * (self.PG_PIECE_H + self.PG_GAP),
                    self.PG_PIECE_W,
                    self.PG_PIECE_H,
                )
                for c in range(self.BOARD_SIZE)
            ]
            for r in range(self.BOARD_SIZE)
        ]
        self.pg_board[0][4].score_text = str(-1)
        self.pg_board[0][4].score_color = (255, 0, 0)
        self.pg_board[4][4].score_text = str(1)
        self.pg_board[4][4].score_color = (0, 255, 0)

    def _init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.PG_WINDOW_W, self.PG_WINDOW_H))
        pygame.display.set_caption("GridWorld")
        self._init_assets()

    def _init_offscreen(self):
        pygame.init()
        self._offscreen = pygame.Surface((self.PG_WINDOW_W, self.PG_WINDOW_H))
        self._init_assets()

    def _wait_for_human_click(self, mask=None):
        if self.render_mode != "human":
            return

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.last_action = Action.LEFT
                        return Action.LEFT.value
                    elif event.key == pygame.K_RIGHT:
                        self.last_action = Action.RIGHT
                        return Action.RIGHT.value
                    elif event.key == pygame.K_UP:
                        self.last_action = Action.UP
                        return Action.UP.value
                    elif event.key == pygame.K_DOWN:
                        self.last_action = Action.DOWN
                        return Action.DOWN.value

    def __str__(self):
        return "GridWorldEnv"