import pygame
import numpy as np
from deeprl_5iabd.config import settings
from deeprl_5iabd.envs.base_env import BaseEnv
from deeprl_5iabd.agents.random_agent import RandomPlayer

class QuartoButton:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.image = None

    def draw(self, screen):
        if self.image:
            screen.blit(self.image, self.rect)
        else:
            pygame.draw.rect(screen, (200, 200, 200), self.rect)

    def is_clicked(self, event):
        return event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos)

class QuartoEnv(BaseEnv):
    BOARD_SIZE    = 4
    NUM_PIECES    = 16
    NUM_ATTRS     = 4

    PG_PIECE_W    = 120
    PG_PIECE_H    = 168
    PG_GAP        = 4
    PG_WINDOW_W   = (PG_PIECE_W + PG_GAP) * (BOARD_SIZE * 2) + PG_PIECE_W
    PG_WINDOW_H   = (PG_PIECE_H + PG_GAP) * BOARD_SIZE + PG_PIECE_H

    # Toutes les combinaisons gagnantes (lignes, colonnes, diagonales)
    WIN_PATTERNS = [
        [(0,0),(0,1),(0,2),(0,3)], [(1,0),(1,1),(1,2),(1,3)],  # lignes
        [(2,0),(2,1),(2,2),(2,3)], [(3,0),(3,1),(3,2),(3,3)],
        [(0,0),(1,0),(2,0),(3,0)], [(0,1),(1,1),(2,1),(3,1)],  # colonnes
        [(0,2),(1,2),(2,2),(3,2)], [(0,3),(1,3),(2,3),(3,3)],
        [(0,0),(1,1),(2,2),(3,3)], [(0,3),(1,2),(2,1),(3,0)],  # diagonales
    ]

    # Les 16 pièces encodées en binaire : color, size, shape, fill
    ALL_PIECES = np.array([
        [[1,1,1,1], [1,0,0,0], [1,0,0,1], [1,1,0,0]],
        [[1,1,0,1], [1,0,1,0], [1,0,1,1], [1,1,1,0]],
        [[0,1,1,1], [0,0,0,0], [0,0,0,1], [0,1,0,0]],
        [[0,1,0,1], [0,0,1,0], [0,0,1,1], [0,1,1,0]],
    ])

    def __init__(self):
        super().__init__("quarto")
        self.reset()
        self._pygame_ready = False

    def reset(self) -> None:
        self.available = self.ALL_PIECES.copy()
        self.board = np.full((4, 4, 4), -1)
        self.selected = np.full(4, -1)                 # pièce en cours de placement
        self.selecting = True                          # True=choisir pièce, False=placer
        self.player = 0                                # 0 = joueur 0, 1 = joueur 1

    def step(self, action: int) -> None:
        r, c = divmod(action if self.selecting else action - self.NUM_PIECES, self.BOARD_SIZE)

        if self.selecting:
            self.selected = self.available[r, c].copy()
            self.available[r, c] = np.full(4, -1)
            self.player = 0 if self.player == 1 else 1
        else:
            self.board[r, c] = self.selected
            self.selected = np.full(4, -1)

        self.selecting = not self.selecting

    def get_action_space(self) -> list[int]:
        empty = [0] * self.NUM_PIECES

        if self.selecting:
            picks = [0 if self.available[r, c, 0] == -1 else 1 for r in range(4) for c in range(4)]
            return  picks + empty
        else:
            picks = [1 if self.board[r, c, 0] == -1 else 0 for r in range(4) for c in range(4)]
            return  empty + picks

    def get_observation_space(self) -> list[int]:
        return self.selected.tolist() + self.board.flatten().tolist() + self.available.flatten().tolist()

    def is_game_over(self) -> bool:
        for pattern in self.WIN_PATTERNS:
            cells = np.array([self.board[r, c] for r, c in pattern])
            if -1 not in cells and np.any(np.all(cells == cells[0], axis=0)):
                return True
        return bool(np.all(self.available[:, :, 0] == -1))

    def score(self) -> int:
        if np.all(self.available[:, :, 0] == -1):
            return 0
        return 1 if self.player == 0 else -1


    # Pygame rendering
    def render(self) -> None:
        if not self._pygame_ready:
            self._init_pygame()
            self._pygame_ready = True

        self.screen.fill((0, 0, 0))

        font = pygame.font.SysFont(None, 36)
        phase = "choisissez une pièce" if self.selecting else "placez la pièce"
        self.screen.blit(font.render(f"Joueur {self.player} — {phase}", True, (255, 255, 255)), (10, 10))

        for r in range(4):
            for c in range(4):
                self.pg_board[r][c].image    = self._asset(self.board[r, c])
                self.pg_pieces[r][c].image   = self._asset(self.available[r, c])
                self.pg_board[r][c].draw(self.screen)
                self.pg_pieces[r][c].draw(self.screen)

        self.pg_selected.image = self._asset(self.selected)
        self.pg_selected.draw(self.screen)
        pygame.display.flip()

    def _asset(self, piece: np.ndarray):
        return self.pg_assets.get("".join(map(str, piece)))

    def _init_pygame(self):
        pygame.init()
        self.screen    = pygame.display.set_mode((self.PG_WINDOW_W, self.PG_WINDOW_H))
        self.pg_assets = {
            f"{i:04b}": pygame.transform.scale(
                pygame.image.load(f"{settings.games_assets_path}/quarto_assets/{i:04b}.png"),
                (self.PG_PIECE_W, self.PG_PIECE_H)
            )
            for i in range(self.NUM_PIECES)
        }
        self.pg_assets["-1-1-1-1"] = None

        self.pg_board   = [[QuartoButton(c * (self.PG_PIECE_W + self.PG_GAP),
                                         (r + 1) * (self.PG_PIECE_H + self.PG_GAP),
                                         self.PG_PIECE_W, self.PG_PIECE_H)
                            for c in range(4)] for r in range(4)]

        self.pg_pieces  = [[QuartoButton((c + 5) * (self.PG_PIECE_W + self.PG_GAP),
                                         (r + 1) * (self.PG_PIECE_H + self.PG_GAP),
                                         self.PG_PIECE_W, self.PG_PIECE_H)
                            for c in range(4)] for r in range(4)]

        self.pg_selected = QuartoButton(self.PG_WINDOW_W - self.PG_PIECE_W, 0,
                                        self.PG_PIECE_W, self.PG_PIECE_H)

    # Game modes
    def play_vs_random(self):
        agent   = RandomPlayer(action_dim=self.NUM_PIECES * 2)
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
                for r in range(4):
                    for c in range(4):
                        grid = self.pg_pieces if self.selecting else self.pg_board
                        if grid[r][c].is_clicked(event):
                            offset = 0 if self.selecting else self.NUM_PIECES
                            return r * 4 + c + offset

def main():
    QuartoEnv().play_vs_random()

if __name__ == "__main__":
    main()
