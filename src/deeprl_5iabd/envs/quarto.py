import pygame
import time
import numpy as np
import torch
import random
from deeprl_5iabd.config import settings
from deeprl_5iabd.envs.base_env import BaseEnv
from deeprl_5iabd.agents.random_agent import RandomPlayer
from deeprl_5iabd.helper import ImageButton

class QuartoEnv(BaseEnv):
    """
    Jeu de quarto 4x4.
    Le joueur 0 commence toujours.
    Actions space [0-15]: placer pièce, 16-31: choisir une pièce
    Description space [0..132]: selected(4) + board(16*4) + available(16*4) 
    Récompenses : -1.0 si le joueur 1 gagne, 0.0 si match nul, 1.0 si le joueur 0 gagne.
    """

    PG_PIECE_W    = 120
    PG_PIECE_H    = 168
    PG_GAP        = 4
    PG_WINDOW_W   = (PG_PIECE_W + PG_GAP) * (4 * 2) + PG_PIECE_W
    PG_WINDOW_H   = (PG_PIECE_H + PG_GAP) * 4 + PG_PIECE_H

    # Toutes les combinaisons gagnantes (lignes, colonnes, diagonales)
    VICTORY_PATTERNS =  [
        [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], # Lignes horizontales
        [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], # Colonnes verticales
        [0, 5, 10, 15], [3, 6, 9, 12] # Diagonales
    ]

    # Les 16 pièces encodées en binaire : color, size, shape, fill
    ALL_PIECES = torch.tensor([
        1, 1, 1, 1,  1, 0, 0, 0,  1, 0, 0, 1,  1, 1, 0, 0,
        1, 1, 0, 1,  1, 0, 1, 0,  1, 0, 1, 1,  1, 1, 1, 0,
        0, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 1,  0, 1, 0, 0,
        0, 1, 0, 1,  0, 0, 1, 0,  0, 0, 1, 1,  0, 1, 1, 0,
    ], device=settings.device)

    def __init__(self):
        super().__init__("quarto")
        self.selected = torch.full((4,), -1, device=settings.device)
        self.board = torch.full((16 * 4,), -1,  device=settings.device)
        self.available_ps = torch.full((16 * 4,), -1, device=settings.device)
        self._action_mask_buffer = torch.zeros((16 + 16),  device=settings.device)
        self._obs_mask_buffer = torch.zeros(4 + 16 * 4 + 16 * 4,  device=settings.device)
        self._pygame_ready = False
        self.reset()

    def reset(self) -> None:
        self.selected = self.selected.fill_(-1)
        self.board = self.board.fill_(-1)
        self.available_ps = self.available_ps.copy_(self.ALL_PIECES)
        self.selecting = True
        self.is_game_over = False
        self.score = 0.0
        self._pieces_left = 16
        self.agent_player = random.choice([0, 1])
        self.current_player = 0

    def determinize(self) -> BaseEnv:
        new_env = QuartoEnv()
        new_env.selected = new_env.selected.copy_(self.selected)
        new_env.board = new_env.board.copy_(self.board)
        new_env.available_ps = new_env.available_ps.copy_(self.available_ps)
        new_env.selecting = self.selecting
        new_env.agent_player = self.agent_player
        new_env._pygame_ready = self._pygame_ready
        return new_env

    def step(self, action: int) -> None:
        if self.is_game_over:
            return
        if self.selecting:
            action -= 16
            action *= 4
            self.selected = self.selected.copy_(self.available_ps[action:action+4])
            self.available_ps[action:action+4] = -1
            self.current_player = 0 if self.current_player == 1 else 1
        else:
            action *= 4
            self.board[action:action+4] = self.selected
            self.selected = self.selected.fill_(-1)
            self._pieces_left -= 1

            # is game over
            for (a, b, c, d) in self.VICTORY_PATTERNS:
                if (self.board[a*4] == -1 ):
                    continue
                for i in range(4):
                    if self.board[a*4+i] == self.board[b*4+i] == self.board[c*4+i] == self.board[d*4+i]:
                        self.is_game_over = True
                        if self.current_player == self.agent_player:
                            self.score = 1.0
                        else:
                            self.score = -1.0
                        break

            if self._pieces_left == 0:
                self.is_game_over = True
                self.score = 0.0

        self.selecting = not self.selecting

    def get_action_mask(self) -> torch.Tensor:
        if self.selecting:
            self._action_mask_buffer[:16].zero_()
            self._action_mask_buffer[16:] = (self.available_ps[0::4] != -1).float()
        else:
            self._action_mask_buffer[16:].zero_()
            self._action_mask_buffer[:16] = (self.board[0::4] == -1).float()
        return self._action_mask_buffer

    def get_observation_mask(self) -> torch.Tensor:
        self._obs_mask_buffer[0:4] = self.selected
        self._obs_mask_buffer[4:4+16*4] = self.board
        self._obs_mask_buffer[4+16*4:] = self.available_ps
        return self._obs_mask_buffer

    def render(self) -> None:
        def _asset(piece: np.ndarray):
            return self.pg_assets.get(f"{piece[0]}{piece[1]}{piece[2]}{piece[3]}")

        if not self._pygame_ready:
            self._init_pygame()
            self._pygame_ready = True

        self.screen.fill((0, 0, 0))

        for i in range(16):
            r,c = divmod(i, 4)
            self.pg_board[r][c].image = _asset(self.board[i*4:i*4+4])
            self.pg_pieces[r][c].image = _asset(self.available_ps[i*4:i*4+4])
            self.pg_board[r][c].draw(self.screen)
            self.pg_pieces[r][c].draw(self.screen)
        self.pg_selected.image = _asset(self.selected)
        self.pg_selected.draw(self.screen)

        if (self.is_game_over):
            font = pygame.font.SysFont(None, 64)
            text = f"Partie terminée joueur {self.agent_player} a gagné"
            self.screen.blit(font.render(text, True, (255, 255, 255)), (self.PG_WINDOW_W/2 - font.size(text)[0]/2, self.PG_WINDOW_H/2 - font.size(text)[1]/2))
        else:
            font = pygame.font.SysFont(None, 36)
            text = "choisissez une pièce" if self.selecting else "placez la pièce"
            self.screen.blit(font.render(f"Joueur {self.current_player} — {text}", True, (255, 255, 255)), (10, 10))

        pygame.display.flip()

    def _init_pygame(self):
        pygame.init()
        self.screen    = pygame.display.set_mode((self.PG_WINDOW_W, self.PG_WINDOW_H))
        self.pg_assets = {
            f"{i:04b}": pygame.transform.scale(
                pygame.image.load(f"{settings.quarto_assets_path}/{i:04b}.png"),
                (self.PG_PIECE_W, self.PG_PIECE_H)
            )
            for i in range(16)
        }
        self.pg_assets["-1-1-1-1"] = None

        self.pg_board   = [[ImageButton(c * (self.PG_PIECE_W + self.PG_GAP),
                                         (r + 1) * (self.PG_PIECE_H + self.PG_GAP),
                                         self.PG_PIECE_W, self.PG_PIECE_H)
                            for c in range(4)] for r in range(4)]

        self.pg_pieces  = [[ImageButton((c + 5) * (self.PG_PIECE_W + self.PG_GAP),
                                         (r + 1) * (self.PG_PIECE_H + self.PG_GAP),
                                         self.PG_PIECE_W, self.PG_PIECE_H)
                            for c in range(4)] for r in range(4)]

        self.pg_selected = ImageButton(self.PG_WINDOW_W - self.PG_PIECE_W, 0,
                                        self.PG_PIECE_W, self.PG_PIECE_H)

    def humain_vs_random(self):
        agent  = RandomPlayer(action_dim=16 * 2)
        self.render()

        while 1:
            if self.current_player == self.agent_player:
                action = int(np.argmax(agent.forward(x=None, mask=self.get_action_mask())))
            else:
                action = self._wait_for_human_click()

            self.step(action)
            self.render()

    def humain_vs_humain(self):
        self.render()

        while 1:
            if self.current_player == self.agent_player:
                action = self._wait_for_human_click()
            else:
                action = self._wait_for_human_click()
            self.step(action)
            self.render()

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
                            offset = 16 if self.selecting else 0
                            return r * 4 + c + offset

def main():
    QuartoEnv().humain_vs_humain()

if __name__ == "__main__":
    main()
