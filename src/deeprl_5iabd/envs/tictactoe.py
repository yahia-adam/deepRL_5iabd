import pygame
import numpy as np
from deeprl_5iabd.envs.base_env import ModelBasedEnv
from deeprl_5iabd.config import settings
from deeprl_5iabd.helper import ImageButton
from deeprl_5iabd.agents.random_agent import RandomPlayer

class TicTacToe(ModelBasedEnv):
    """
    Jeu de morpion 3x3.
    Récompenses : -1.0 si le joueur 1 gagne, 0.0 si match nul, 1.0 si le joueur 0 gagne.
    Le joueur 0 est représenté par des O et le joueur 1 par des X.
    Le joueur 0 commence toujours.
    Actions [0..8]: 0=haut-gauche, 1=haut-milieu, ....
    """

    BOARD_SIZE = 3

    TEXT_H = 40
    PG_GAP = 5

    PG_PIECE_W = 250
    PG_PIECE_H = 250
    PG_WINDOW_W = (PG_PIECE_W + PG_GAP) * BOARD_SIZE
    PG_WINDOW_H = (PG_PIECE_H + PG_GAP) * BOARD_SIZE + TEXT_H

    # Toutes les combinaisons gagnantes
    WIN_PATTERNS = [
        # lignes
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],

        # colonnes
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],

        # diagonales
        [0, 4, 8],
        [2, 4, 6],
    ]

    def __init__(self):
        super().__init__("TicTacToe")
        self.reset()
        self.A = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.R = [-1.0, 0.0, 1.0]
        self._create_p()
        self._pygame_ready = False

    def num_states(self):
        return 3**9

    def num_actions(self):
        return 9

    def num_rewards(self):
        return 3

    def state_id(self, state) -> int:
        state_normalized = np.array(state) + 1
        powers_of_3 = 3 ** np.arange(9)
        return int(np.dot(state_normalized, powers_of_3))

    def determinize(self):
        new_env = TicTacToe()
        new_env.board = self.board.copy()
        new_env.player = self.player
        return new_env

    def reset(self) -> None:
        self.board = np.array([-1]*9)
        self.player = 0

    def is_game_over(self) -> bool:
        if self.score() != 0:
            return True
        if np.all(self.board != -1):
            return True
        return False

    def get_observation_space(self) -> np.ndarray:
        return self.board.tolist()

    def get_action_space(self):
        return (self.board == -1).astype(int)

    def step(self, action: int) -> None:
        self.board[action] = self.player
        self.player = 0 if self.player == 1 else 1

    def score(self) -> float:
        for pattern in self.WIN_PATTERNS:
            a,b,c = pattern
            if self.board[a] == self.board[b] == self.board[c] and self.board[a] != -1:
                return self.R[0] if self.board[a] == 1 else self.R[2]

        return self.R[1]

    def render(self) -> None:
        if not self._pygame_ready:
            self._init_pygame()
            self._pygame_ready = True

        self.screen.fill((0, 0, 0))

        font = pygame.font.SysFont(None, 36)
        self.screen.blit(font.render(f"Joueur {self.player} jouer", True, (255, 255, 255)), (10, 10))

        for r in range(3):
            for c in range(3):
                if self.board[r * self.BOARD_SIZE + c] == 0:
                    self.pg_board[r][c].image = self.pg_assets[0]
                elif self.board[r * self.BOARD_SIZE + c] == 1:
                    self.pg_board[r][c].image = self.pg_assets[1]
                self.pg_board[r][c].draw(self.screen)

        pygame.display.flip()

    def _create_p(self):
        self.p = {}
        self._explore_from_state()

    def _explore_from_state(self):
        pass
        # s_id = self.state_id(self.board)
        # if s_id in self.p:
        #     return

        # self.p[s_id] = {}
        # over = self.is_game_over()
        # mask = self.get_action_space()

        # for a in range(9):
        #     if over or mask[a] == 0:
        #         # On définit l'issue d'une action impossible ou finie
        #         self.p[s_id][a] = [(1.0, s_id, 0.0, True)]
        #     else:
        #         # 1. Sauvegarde l'état du joueur avant le coup
        #         prev_player = self.player

        #         # 2. Joue le coup (modifie self.board et self.player)
        #         self.step(a)

        #         # 3. Enregistre les résultats
        #         next_s_id = self.state_id(self.board)
        #         res_reward = self.score()
        #         res_done = self.is_game_over()
        #         self.p[s_id][a] = [(1.0, next_s_id, res_reward, res_done)]

        #         # 4. Explore la suite si ce n'est pas fini
        #         if not res_done:
        #             self._explore_from_state()

        #         # 5. BACKTRACK : On remet le plateau ET le joueur à l'état initial
        #         self.board[a] = -1
        #         self.player = prev_player

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

    def humain_vs_random(self):
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
    env.humain_vs_random()