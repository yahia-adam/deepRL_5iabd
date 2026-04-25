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
    LEFT = 0
    RIGHT = 1


class LineWorldEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    BOARD_SIZE = 5

    PG_PIECE_W = 250
    PG_PIECE_H = 250
    PG_GAP = 5

    PG_WINDOW_W = (PG_PIECE_W + PG_GAP) * BOARD_SIZE
    PG_WINDOW_H = (PG_PIECE_H + PG_GAP)

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.screen = None
        self._offscreen = None
        self.last_action = 1
        self.agent_pos = 2

        if self.render_mode == "human":
            self._init_pygame()
        elif self.render_mode == "rgb_array":
            self._init_offscreen()

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self._action_mask_buffer = np.ones(self.action_space.n, dtype=np.int8)

        self.board = np.zeros(self.observation_space.shape[0], dtype=np.float32)

        self.current_player = Player.PLAYER_1
        self.agent_player = Player.PLAYER_1
        self.is_multi_player = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.board[:] = 0
        self.agent_pos = 2
        self.last_action = 1
        self.board[self.agent_pos] = 1
        return self._get_obs(), {}

    def determinize(self):
        env = LineWorldEnv()
        env.board[:] = self.board.copy()
        env.agent_pos = self.agent_pos
        env.last_action = self.last_action
        env.current_player = self.current_player
        env.agent_player = self.agent_player
        env.is_multi_player = self.is_multi_player

        return env

    def step(self, action):
        self.board[self.agent_pos] = 0

        if action == Action.LEFT.value:
            self.agent_pos = max(0, self.agent_pos - 1)
            self.last_action = Action.LEFT.value
        elif action == Action.RIGHT.value:
            self.agent_pos = min(4, self.agent_pos + 1)
            self.last_action = Action.RIGHT.value
        self.board[self.agent_pos] = 1

        terminated = (self.agent_pos == 0) or (self.agent_pos == 4)

        if self.agent_pos == 0:
            reward = -1.0
        elif self.agent_pos == 4:
            reward = 1.0
        else:
            reward = 0.0

        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        if self.render_mode not in ("human", "rgb_array"):
            return

        surface = self.screen if self.render_mode == "human" else self._offscreen
        surface.fill((30, 30, 30))

        for i in range(self.BOARD_SIZE):
            self.pg_board[i].image = None
            if i == self.agent_pos:
                self.pg_board[i].image = self.pg_assets[self.last_action]
            self.pg_board[i].draw(surface)

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
        return self.agent_pos

    def _init_assets(self):
        self.pg_assets = [
            pygame.transform.scale(
                pygame.image.load(f"{settings.line_world_assets_path}/{i}.png"),
                (self.PG_PIECE_W, self.PG_PIECE_H)
            )
            for i in range(0, 2)
        ]

        self.pg_board = [
            ImageButton(
                x=c * self.PG_PIECE_W + c * self.PG_GAP,
                y=0,
                width=self.PG_PIECE_W,
                height=self.PG_PIECE_H
            )
            for c in range(self.BOARD_SIZE)
        ]
        self.pg_board[0].score_text = str(-1)
        self.pg_board[0].score_color = (255, 0, 0)
        self.pg_board[4].score_text = str(1)
        self.pg_board[4].score_color = (0, 255, 0)

    def _init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.PG_WINDOW_W, self.PG_WINDOW_H))
        pygame.display.set_caption("LineWorld")
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
                        return Action.LEFT.value
                    elif event.key == pygame.K_RIGHT:
                        return Action.RIGHT.value

    def __str__(self):
        return "LineWorldEnv"