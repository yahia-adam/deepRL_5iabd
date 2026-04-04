import pygame
import torch
import random
import numpy as np
import quarto_cpp

from deeprl_5iabd.config import settings
from deeprl_5iabd.envs.base_env import BaseEnv
from deeprl_5iabd.agents.random_agent import RandomPlayer
from deeprl_5iabd.helper import ImageButton

class QuartoEnv(BaseEnv):
    PG_PIECE_W    = 120
    PG_PIECE_H    = 168
    PG_GAP        = 4
    PG_WINDOW_W   = (PG_PIECE_W + PG_GAP) * (4 * 2) + PG_PIECE_W
    PG_WINDOW_H   = (PG_PIECE_H + PG_GAP) * 4 + PG_PIECE_H

    def __init__(self):
        super().__init__("quarto")
        self.cpp_env = quarto_cpp.QuartoEnvCpp()
        self._pygame_ready = False

    @property
    def board(self): return self.cpp_env.board

    @property
    def selected(self): return self.cpp_env.selected_piece

    @property
    def available_ps(self): return self.cpp_env.available_pieces

    @property
    def is_game_over(self): return self.cpp_env.is_game_over

    @property
    def selecting(self): return self.cpp_env.selecting

    @property
    def current_player(self): return self.cpp_env.current_player

    @property
    def agent_player(self): return self.cpp_env.agent_player

    @property
    def score(self): return self.cpp_env.score

    def reset(self) -> None:
        self.cpp_env.reset()

    def step(self, action: int) -> None:
        self.cpp_env.step(action)

    def get_action_mask(self) -> torch.Tensor:
        mask_list = self.cpp_env.get_action_mask()
        return torch.tensor(mask_list, dtype=torch.float32, device=settings.device)

    def get_observation_mask(self) -> torch.Tensor:
        obs_list = self.cpp_env.get_observation_mask()
        return torch.tensor(obs_list, dtype=torch.float32, device=settings.device)

    def determinize(self) -> BaseEnv:
        new_env = QuartoEnv()
        # Copie profonde des états vers le nouveau wrapper C++
        new_env.cpp_env.board = self.cpp_env.board
        new_env.cpp_env.selected_piece = self.cpp_env.selected_piece
        new_env.cpp_env.available_pieces = self.cpp_env.available_pieces
        new_env.cpp_env.selecting = self.cpp_env.selecting
        new_env.cpp_env.agent_player = self.cpp_env.agent_player
        new_env.cpp_env.current_player = self.cpp_env.current_player
        new_env.cpp_env.is_game_over = self.cpp_env.is_game_over
        new_env.cpp_env.score = self.cpp_env.score
        new_env._pygame_ready = self._pygame_ready
        return new_env

    # --- Rendu Pygame ---
    def render(self) -> None:
        def _asset(piece):
            return self.pg_assets.get(f"{piece[0]}{piece[1]}{piece[2]}{piece[3]}")

        if not self._pygame_ready:
            self._init_pygame()
            self._pygame_ready = True

        self.screen.fill((0, 0, 0))

        for i in range(16):
            r, c = divmod(i, 4)
            self.pg_board[r][c].image = _asset(self.board[i*4 : i*4+4])
            self.pg_pieces[r][c].image = _asset(self.available_ps[i*4 : i*4+4])
            
            self.pg_board[r][c].draw(self.screen)
            self.pg_pieces[r][c].draw(self.screen)
            
        self.pg_selected.image = _asset(self.selected)
        self.pg_selected.draw(self.screen)

        if self.is_game_over:
            font = pygame.font.SysFont(None, 64)
            text = f"Partie terminee joueur {self.current_player} a gagne"
            self.screen.blit(font.render(text, True, (255, 255, 255)), 
                             (self.PG_WINDOW_W/2 - font.size(text)[0]/2, 
                              self.PG_WINDOW_H/2 - font.size(text)[1]/2))
        else:
            font = pygame.font.SysFont(None, 36)
            text = "choisissez une piece" if self.selecting else "placez la piece"
            self.screen.blit(font.render(f"Joueur {self.current_player} - {text}", True, (255, 255, 255)), (10, 10))

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
                self.step(action)
            else:
                action = self._wait_for_human_click()
                self.step(action)

            self.render()

    def humain_vs_humain(self):
        self.render()
        while 1:
            if self.current_player == self.agent_player:
                action = self._wait_for_human_click()
                self.step(action)
            else:
                action = self._wait_for_human_click()
                self.step(action)
            print("step", step)
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
    QuartoEnv().humain_vs_random()

if __name__ == "__main__":
    main()