import pygame
import numpy as np
from deeprl_5iabd.envs.base_env import BaseEnv
from deeprl_5iabd.config import settings
from deeprl_5iabd.envs.quarto import QuartoButton
from deeprl_5iabd.agents.random_agent import RandomPlayer

class TicTacToe(BaseEnv):

    BOARD_SIZE = 3

    PG_PIECE_W = 120
    PG_PIECE_H = 120
    PG_GAP = 5

    TEXT_H = 40

    PG_WINDOW_W = (PG_PIECE_W + PG_GAP) * BOARD_SIZE
    PG_WINDOW_H = (PG_PIECE_H + PG_GAP) * BOARD_SIZE + TEXT_H

    # Toutes les combinaisons gagnantes (lignes, colonnes, diagonales)
    WIN_PATTERNS = [
        [(0,0),(0,1),(0,2)],
        [(1,0),(1,1),(1,2)],
        [(2,0),(2,1),(2,2)],

        [(0,0),(1,0),(2,0)],
        [(0,1),(1,1),(2,1)],
        [(0,2),(1,2),(2,2)],

        [(0,0),(1,1),(2,2)],
        [(0,2),(1,1),(2,0)],
    ]

    def __init__(self):
        super().__init__("TicTacToe")
        self.reset()
        self._pygame_ready = False

    def reset(self) -> None:
        self.board = np.full((3, 3), -1)
        self.player = 0

    def is_game_over(self) -> bool:
        if self.score() != 0:
            return True
        if np.all(self.board != -1):
            return True
        return False

    def get_observation_space(self) -> np.ndarray:
        return self.board.flatten().tolist()

    def get_action_space(self):
        action_space = []
        for r in range(3):
            for c in range(3):
                if self.board[r, c] == -1:
                    action_space.append(1)
                else:
                    action_space.append(0)
        return action_space

    def step(self, action: int) -> None:
        r, c = divmod(action, 3)
        self.board[r, c] = self.player
        self.player = 0 if self.player == 1 else 1

    def score(self) -> int:
        for pattern in self.WIN_PATTERNS:
            cells = np.array([self.board[r, c] for r, c in pattern])
            if -1 not in cells and np.all(cells == cells[0]):
                return -1 if cells[0] == 1 else 1

        return 0

    def render(self) -> None:
        if not self._pygame_ready:
            self._init_pygame()
            self._pygame_ready = True

        self.screen.fill((0, 0, 0))

        font = pygame.font.SysFont(None, 36)
        self.screen.blit(font.render(f"Joueur {self.player} jouer", True, (255, 255, 255)), (10, 10))

        for r in range(3):
            for c in range(3):
                if self.board[r, c] == 0:
                    self.pg_board[r][c].image = self.pg_assets[0]
                elif self.board[r, c] == 1:
                    self.pg_board[r][c].image = self.pg_assets[1]
                self.pg_board[r][c].draw(self.screen)

        pygame.display.flip()


    def _init_pygame(self) -> None:
        pygame.init()

        self.screen = pygame.display.set_mode(
            (self.PG_WINDOW_W, self.PG_WINDOW_H)
        )

        pygame.display.set_caption("TicTacToe")

        self.pg_assets = [
            pygame.transform.scale(
                pygame.image.load(
                    f"{settings.games_assets_path}/tictactoe_assets/0.jpg"
                ),
                (self.PG_PIECE_W, self.PG_PIECE_H),
            ),
            pygame.transform.scale(
                pygame.image.load(
                    f"{settings.games_assets_path}/tictactoe_assets/1.jpg"
                ),
                (self.PG_PIECE_W, self.PG_PIECE_H),
            ),
        ]

        self.pg_board = [
            [
                QuartoButton(
                    c * (self.PG_PIECE_W + self.PG_GAP),
                    r * (self.PG_PIECE_H + self.PG_GAP) + self.TEXT_H,
                    self.PG_PIECE_W,
                    self.PG_PIECE_H,
                )
                for c in range(3)
            ]
            for r in range(3)
        ]

  # Game modes
    def play_vs_random(self):
        agent   = RandomPlayer(action_dim=len(self.get_action_space()))
        running = True
        self.render()

        while running:
            if self.player == 0:
                action = int(np.argmax(agent.forward(x=None, mask=self.get_action_space())))
            else:
                action = self._wait_for_human_click()
            
            self.step(action)
            self.render()
            running = not self.is_game_over()

        self.render()
        print({0: "Match nul", 1: "Random a gagné", -1: "Vous avez gagné"}[self.score()])

    def _wait_for_human_click(self) -> int:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit
                for r in range(3):
                    for c in range(3):
                        if self.pg_board[r][c].is_clicked(event):
                            return r * 3 + c

if __name__ == "__main__":
    env = TicTacToe()
    env.play_vs_random()