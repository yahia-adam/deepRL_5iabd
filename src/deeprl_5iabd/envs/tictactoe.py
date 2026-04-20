import time
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from deeprl_5iabd.helper import Player
from deeprl_5iabd.config import settings
from deeprl_5iabd.helper import ImageButton

class TicTacToeEnv(gym.Env):
    """
    Environnement TicTacToe (Morpion 3x3).
    - Joueur 0 (O) : commence, cherche à maximiser (récompense +1.0 si victoire)
    - Joueur 1 (X) : cherche à minimiser (récompense -1.0 si victoire)
    - Match nul : récompense 0.0
    - Grille vide = -1
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    # Combinaisons gagnantes pré-calculées pour des perfs maximales
    WIN_PATTERNS = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], # Lignes
        [0, 3, 6], [1, 4, 7], [2, 5, 8], # Colonnes
        [0, 4, 8], [2, 4, 6]             # Diagonales
    ]
    BOARD_SIZE = 3

    TEXT_H = 40
    PG_GAP = 5

    PG_PIECE_W = 250
    PG_PIECE_H = 250
    PG_WINDOW_W = (PG_PIECE_W + PG_GAP) * BOARD_SIZE
    PG_WINDOW_H = (PG_PIECE_H + PG_GAP) * BOARD_SIZE + TEXT_H

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        self.board = np.full(9, -1, dtype=np.float32)

        self.action_space = spaces.Discrete(len(self.board), dtype=np.int8) 
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1+len(self.board),), dtype=np.float32)

        self._obs_buffer = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        self._action_mask_buffer = np.zeros(self.action_space.n, dtype=np.int8)

        self._pygame_ready = False
        self.screen = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.board[:] = -1
        self.current_player = Player.PLAYER_1
        self.agent_player = np.random.choice([Player.PLAYER_1, Player.PLAYER_2])
        self.count_step = 0

        return self._get_obs(), {}

    def step(self, action):
        self.board[action] = self.current_player.value
        self.count_step += 1

        terminated = False
        truncated = False
        reward = 0.0

        if self._check_win():
            terminated = True
            reward = 1.0 if self.current_player == self.agent_player else -1.0
        elif self.count_step == 9:
            terminated = True
            reward = 0.0

        self.current_player = Player.PLAYER_2 if self.current_player == Player.PLAYER_1 else Player.PLAYER_1
        return self._get_obs(), reward, terminated, False, {}

    def render(self) -> None:
        if self.render_mode != "human":
            return

        if not self._pygame_ready:
            self._init_pygame()
            self._pygame_ready = True

        self.screen.fill((0, 0, 0))

        font = pygame.font.SysFont(None, 36)
        self.screen.blit(font.render(f"Joueur {self.current_player} jouer", True, (255, 255, 255)), (10, 10))

        for r in range(3):
            for c in range(3):
                if self.board[r * 3 + c] == Player.PLAYER_1.value:
                    self.pg_board[r][c].image = self.pg_assets[0]
                if self.board[r * 3 + c] == Player.PLAYER_2.value:
                    self.pg_board[r][c].image = self.pg_assets[1]
                self.pg_board[r][c].draw(self.screen)

        pygame.display.flip()


    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def _init_pygame(self) -> None:
        pygame.init()

        self.screen = pygame.display.set_mode(
            (self.PG_WINDOW_W, self.PG_WINDOW_H)
        )

        pygame.display.set_caption("TicTacToe")

        self.pg_assets = [
            pygame.transform.scale(
                pygame.image.load(
                    f"{settings.tictactoe_assets_path}/0.png"
                ),
                (self.PG_PIECE_W, self.PG_PIECE_H),
            ),
            pygame.transform.scale(
                pygame.image.load(
                    f"{settings.tictactoe_assets_path}/1.png"
                ),
                (self.PG_PIECE_W, self.PG_PIECE_H),
            ),
        ]

        self.pg_board = [
            [
                ImageButton(
                    c * (self.PG_PIECE_W + self.PG_GAP),
                    r * (self.PG_PIECE_H + self.PG_GAP) + self.TEXT_H,
                    self.PG_PIECE_W,
                    self.PG_PIECE_H,
                )
                for c in range(3)
            ]
            for r in range(3)
        ]


    def _check_win(self):
        for p in self.WIN_PATTERNS:
            if self.board[p[0]] == self.board[p[1]] == self.board[p[2]] != -1:
                return True
        return False

    def _get_obs(self):
        np.concatenate(([self.current_player], self.board), out=self._obs_buffer)
        return self._obs_buffer


    def _get_action_mask(self):
        return (self.board == -1).astype(np.int8)


    def _wait_for_human_click(self, mask) -> int:
        if (self._pygame_ready == False):
            self._init_pygame()
            self._pygame_ready = True
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit
                for r in range(3):
                    for c in range(3):
                        if self.pg_board[r][c].is_clicked(event):
                            if mask[r * 3 + c] == 1:
                                return r * 3 + c
