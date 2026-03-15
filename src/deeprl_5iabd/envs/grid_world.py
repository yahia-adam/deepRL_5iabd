import pygame
import numpy as np
from deeprl_5iabd.envs.base_env import BaseEnv
from deeprl_5iabd.config import settings
from deeprl_5iabd.helper import ImageButton
from deeprl_5iabd.agents.random_agent import RandomPlayer

class GridWorld(BaseEnv):
    BOARD_SIZE = 5

    PG_PIECE_W = 150
    PG_PIECE_H = 150
    PG_GAP = 5

    PG_WINDOW_W = (PG_PIECE_W + PG_GAP) * BOARD_SIZE
    PG_WINDOW_H = (PG_PIECE_H + PG_GAP) * BOARD_SIZE

    def __init__(self):
        super().__init__("GridWorld")
        self.reset()
        self._pygame_initialized = False

    def reset(self) -> None:
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE))
        self.agent_pos = (0, 0)
        self.board[self.agent_pos] = 1

    def is_game_over(self) -> bool:
        return self.agent_pos == (4, 4) or self.agent_pos == (0, 4)

    def get_observation_space(self):
        return self.board.flatten().tolist()

    def get_action_space(self) -> list[int]:
        picks = np.ones(4)
        if self.agent_pos[0] == 0:
            picks[1] = 0
        if self.agent_pos[1] == 0:
            picks[3] = 0
        if self.agent_pos[0] == self.BOARD_SIZE - 1:
            picks[0] = 0
        if self.agent_pos[1] == self.BOARD_SIZE - 1:
            picks[2] = 0
        return picks.tolist()

    def step(self, action: int) -> None:
        """0: bas, 1: haut, 2: droite, 3: gauche"""
        self.board[self.agent_pos] = 0
        if action == 0:
            self.agent_pos = (self.agent_pos[0] + 1, self.agent_pos[1])
        elif action == 1:
            self.agent_pos = (self.agent_pos[0] - 1, self.agent_pos[1])
        elif action == 2:
            self.agent_pos = (self.agent_pos[0], self.agent_pos[1] + 1)
        elif action == 3:
            self.agent_pos = (self.agent_pos[0], self.agent_pos[1] - 1)
        self.board[self.agent_pos] = 1

    def score(self) -> int:
        if self.agent_pos == (4, 4):
            return -3
        elif self.agent_pos == (0, 4):
            return 5
        else:
            return 0

    def render(self) -> None:
        if not self._pygame_initialized:
            self._init_pygame()
            self._pygame_initialized = True
        
        self.screen.fill((30, 30, 30))
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if self.board[r, c] == 1:
                    self.pg_board[r][c].image = self.pg_assets[self.last_action]
                else:
                    self.pg_board[r][c].image = None
                self.pg_board[r][c].draw(self.screen)
        pygame.display.flip()

    def _init_pygame(self) -> None:
        pygame.init()
        self.last_action = 0
        self.screen = pygame.display.set_mode((self.PG_WINDOW_W, self.PG_WINDOW_H))
        pygame.display.set_caption("GridWorld")

        self.pg_assets = [
            pygame.transform.scale(
                pygame.image.load(
                    f"{settings.grid_world_assets_path}/{i}.png"
                ),
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

  # Game modes
    def _play(self):
        self.reset()
        while not self.is_game_over():
            self.render()
            action = None
            actions = self.get_action_space()
            while action is None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            action = 3
                        elif event.key == pygame.K_RIGHT:
                            action = 2
                        elif event.key == pygame.K_UP:
                            action = 1
                        elif event.key == pygame.K_DOWN:
                            action = 0

                        if actions[action] == 1:
                            self.last_action = action
                            self.step(action)
                            self.render()

if __name__ == "__main__":
    env = GridWorld()
    env._play()
