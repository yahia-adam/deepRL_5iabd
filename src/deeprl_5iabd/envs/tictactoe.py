import pygame
import numpy as np
from deeprl_5iabd.envs.model_based_env import ModelBasedEnv
from deeprl_5iabd.config import settings
from deeprl_5iabd.helper import ImageButton
from deeprl_5iabd.agents.random_agent import RandomPlayer

class TicTacToe(ModelBasedEnv):

    BOARD_SIZE = 3

    PG_PIECE_W = 250
    PG_PIECE_H = 250
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
        self.T = []
        self.A = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.R = [-1.0, 0.0, 1.0]
        self.p_matrix = self._create_p_matrix()
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
        return (self.board.flatten() == -1).astype(int)

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

    def _check_terminal(self, board_flat):
        """
        Analyse une grille plate pour déterminer si le jeu est fini.
        Retourne (score, is_over)
        """
        # On replace en 3x3 pour utiliser WIN_PATTERNS
        b = board_flat.reshape(3, 3)
        for pattern in self.WIN_PATTERNS:
            cells = [b[r, c] for r, c in pattern]
            # Si toutes les cellules d'un pattern sont identiques et non vides
            if -1 not in cells and all(x == cells[0] for x in cells):
                # Joueur 0 gagne -> +1, Joueur 1 gagne -> -1
                return (1 if cells[0] == 0 else -1), True
        
        # Match nul : plus de cases vides
        if -1 not in board_flat:
            return 0, True
            
        return 0, False

    def _create_p_matrix(self):
        """Initialise le dictionnaire des dynamiques."""
        self.P = {} 
        initial_state = np.full(9, -1)
        # On lance l'exploration à partir d'une grille vide
        self._explore_from_state(initial_state)
        return self.P

    def _explore_from_state(self, board_flat):
        """Explore récursivement tous les états possibles sans saturer la RAM."""
        s_id = self.state_id(board_flat)
        
        if s_id in self.P:
            return

        self.P[s_id] = {}
        score, over = self._check_terminal(board_flat)

        for a in range(9):
            # Si l'action est impossible (case prise) ou jeu déjà fini
            if over or board_flat[a] != -1:
                # On reste dans le même état (boucle terminale)
                self.P[s_id][a] = [(1.0, s_id, 0.0, True)]
                continue

            # --- Simulation du coup ---
            next_board = board_flat.copy()
            # On compte les pions pour savoir qui doit jouer
            # (Pair = Joueur 0, Impair = Joueur 1)
            nb_pions = np.sum(board_flat != -1)
            current_player = 0 if nb_pions % 2 == 0 else 1
            
            next_board[a] = current_player
            
            # --- Calcul du résultat ---
            next_s_id = self.state_id(next_board)
            reward, is_done = self._check_terminal(next_board)
            
            # Stockage : (probabilité, s_suivant, récompense, fini)
            self.P[s_id][a] = [(1.0, next_s_id, float(reward), is_done)]
            
            # Si le coup n'a pas terminé la partie, on explore la suite
            if not is_done:
                self._explore_from_state(next_board)
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