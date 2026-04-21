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

    BOARD_SIZE = 5

    PG_PIECE_W = 250
    PG_PIECE_H = 250
    PG_GAP = 5

    PG_WINDOW_W = (PG_PIECE_W + PG_GAP) * BOARD_SIZE
    PG_WINDOW_H = (PG_PIECE_H + PG_GAP)

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self.board = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        self._pygame_initialized = False

        self.current_player = Player.PLAYER_1
        self.agent_player = Player.PLAYER_2
        self.is_game_over = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.board[:] = 0
        self.agent_pos = 2
        self.board[self.agent_pos] = 1
        return self.board, {}

    def step(self, action):
        self.board[self.agent_pos] = 0

        if action == Action.LEFT.value:
            self.agent_pos = max(0, self.agent_pos - 1)
        elif action == Action.RIGHT.value:
            self.agent_pos = min(4, self.agent_pos + 1)
        self.board[self.agent_pos] = 1

        terminated = (self.agent_pos == 0) or (self.agent_pos == 4)

        if self.agent_pos == 0:
            reward = -1.0
        elif self.agent_pos == 4:
            reward = 1.0
        else:
            reward = 0.0

        return self.agent_pos, reward, terminated, False, {}

    def render(self) -> None:
        if not self._pygame_initialized:
            self._init_pygame()

        self.screen.fill((30, 30, 30))

        for i in range(self.BOARD_SIZE):
            self.pg_board[i].image = None
            if i == self.agent_pos:
                self.pg_board[i].image = self.pg_assets[self.last_action]
            self.pg_board[i].draw(self.screen)
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def _get_action_mask(self):
        # rien a return
        return

    def _init_pygame(self):
        pygame.init()
        self.last_action = 1
        self.screen = pygame.display.set_mode((self.PG_WINDOW_W, self.PG_WINDOW_H))
        pygame.display.set_caption("LineWorld")
        self._pygame_initialized = True

        self.pg_assets = [
            pygame.transform.scale(
                pygame.image.load(f"{settings.line_world_assets_path}/{i}.png"),
                (self.PG_PIECE_W, self.PG_PIECE_H)
            )
            for i in range(0, 2)
        ]

        self.pg_board   = [ImageButton(x = c * self.PG_PIECE_W + c * self.PG_GAP, y = 0,
                                        width = self.PG_PIECE_W, height = self.PG_PIECE_H)
                            for c in range(self.BOARD_SIZE)]
        self.pg_board[0].score_text = str(-1)
        self.pg_board[0].score_color = (255, 0, 0)
        self.pg_board[4].score_text = str(1)
        self.pg_board[4].score_color = (0, 255, 0)

    def _wait_for_human_click(self, mask=None):
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
