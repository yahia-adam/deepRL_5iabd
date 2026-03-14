import pygame
import numpy as np
from deeprl_5iabd.config import settings
from deeprl_5iabd.envs.base_env import BaseEnv
from deeprl_5iabd.agents.random_agent import RandomPlayer

class QuartoButton:
    def __init__(self, x, y, width, height, image=None, color=(200, 200, 200)):
        self.rect = pygame.Rect(x, y, width, height)
        self.image = image
        self.color = color

    def set_image(self, image):
        self.image = image

    def clear(self):
        self.image = None

    def draw(self, screen):
        if self.image:
            screen.blit(self.image, self.rect)
        else:
            pygame.draw.rect(screen, self.color, self.rect)

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            return self.rect.collidepoint(event.pos)
        return False

class QuartoEnv(BaseEnv):
    # COLOR_IDX = 0;   // light/dark
    # SIZE_IDX = 1;    // large/small
    # SHAPE_IDX = 2;   // square/round
    # FILL_IDX = 3;    // solid/hollow

    NUM_PIECES = 16
    PIECE_ATTRIBUTES = 4
    BOARD_SIZE = NUM_PIECES * PIECE_ATTRIBUTES

    PG_PIECE_HEIGHT = 84*2
    PG_PIECE_WIDTH = 60*2
    PG_GAP = 4
    PG_WINDOW_WIDTH = (PG_PIECE_WIDTH + PG_GAP) * (PIECE_ATTRIBUTES * 2) + PG_PIECE_WIDTH
    PG_WINDOW_HEIGHT = (PG_PIECE_HEIGHT + PG_GAP) * PIECE_ATTRIBUTES + PG_PIECE_HEIGHT

    VICTORY_PATTERNS = [
        [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], # Lignes horizontales
        [0, 4, 8, 12],   [1, 5, 9, 13],   [2, 6, 10, 14],  [3, 7, 11, 15], # Colonnes verticales
        [0, 5, 10, 15], [3, 6, 9, 12] # Diagonales
    ]

    def __init__(self):
        super().__init__("quarto")
        self.reset()
        self._pygame_initialized = False

    def reset(self) -> None:
        self.all_pieces = [
            1, 1, 1, 1,  1, 0, 0, 0,  1, 0, 0, 1,  1, 1, 0, 0,
            1, 1, 0, 1,  1, 0, 1, 0,  1, 0, 1, 1,  1, 1, 1, 0,
            0, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 1,  0, 1, 0, 0,
            0, 1, 0, 1,  0, 0, 1, 0,  0, 0, 1, 1,  0, 1, 1, 0,
        ]
        self.available_pieces = self.all_pieces.copy()
        self.board = [-1 for _ in range(self.NUM_PIECES * self.PIECE_ATTRIBUTES)]
        self.current_player = 0
        self.is_selecting_phase = True
        self.selected_piece = [-1,-1,-1,-1]

    def step(self, action) -> None:
        if self.is_selecting_phase:
            self.selected_piece = self._get_piece(action, self.available_pieces)
            self._update_piece([-1,-1,-1,-1], action, self.available_pieces)
            self.current_player = 1 if self.current_player == 0 else 0
        else:
            action -= self.NUM_PIECES
            self._update_piece(self.selected_piece, action, self.board)
            self.selected_piece = [-1,-1,-1,-1]

        self.is_selecting_phase = not self.is_selecting_phase

    def get_action_space(self) -> list[int]:
        placement_mask = [0 ]* self.NUM_PIECES
        
        if (self.is_selecting_phase):
            return [1 if self.available_pieces[i] != -1 else 0 for i in range(0, self.NUM_PIECES * self.PIECE_ATTRIBUTES, self.PIECE_ATTRIBUTES)] + placement_mask
        else :
            return placement_mask + [1 if self.board[i] == -1 else 0 for i in range(0, self.NUM_PIECES * self.PIECE_ATTRIBUTES, self.PIECE_ATTRIBUTES)]
    
    def get_observation_space(self) -> list[int]:
        return self.selected_piece + self.board + self.available_pieces

    def is_game_over(self) -> bool:
        for e in self.VICTORY_PATTERNS:
            for i in range(self.PIECE_ATTRIBUTES):
                cel_0 = self.board[e[0] * self.PIECE_ATTRIBUTES + i]
                cel_1 = self.board[e[1] * self.PIECE_ATTRIBUTES + i]
                cel_2 = self.board[e[2] * self.PIECE_ATTRIBUTES + i]
                cel_3 = self.board[e[3] * self.PIECE_ATTRIBUTES + i]
                if cel_0 == cel_1 == cel_2 == cel_3 and cel_3 != -1:
                    return True

        if all(val == -1 for val in self.available_pieces):
            return True
        
        return False

    def score(self) -> int:
        is_draw = False if -1 in self.available_pieces else True
        if (is_draw):
            return 0
        elif self.current_player == 0:
            return 1
        else:
            return -1

    def render(self) -> None:
        if not self._pygame_initialized:
            self._init_pygame()
            self._pygame_initialized = True

        self.scrn.fill((0, 0, 0))

        font = pygame.font.SysFont(None, 36)
        txt = f"Joueur {self.current_player} {'choisissez une pièce' if self.is_selecting_phase else 'placez la pièce'}"
        surface = font.render(txt, True, (255, 255, 255))
        self.scrn.blit(surface, (10, self.PG_PIECE_HEIGHT // 2 - surface.get_height() // 2))

        for i in range(self.NUM_PIECES):
            board = self._get_piece(i, self.board)
            piece = self._get_piece(i, self.available_pieces)
            self.pg_board[i].image = self.pg_assets.get(f"{board[0]}{board[1]}{board[2]}{board[3]}", None)
            self.pg_pieces[i].image = self.pg_assets.get(f"{piece[0]}{piece[1]}{piece[2]}{piece[3]}", None)

        self.pg_selected.image = self.pg_assets.get(
            f"{self.selected_piece[0]}{self.selected_piece[1]}{self.selected_piece[2]}{self.selected_piece[3]}", None
        )

        for b in self.pg_board:
            b.draw(self.scrn)
        for p in self.pg_pieces:
            p.draw(self.scrn)
        self.pg_selected.draw(self.scrn)

        pygame.display.flip()

    def _init_pygame(self):
        pygame.init()
        self.scrn = pygame.display.set_mode((self.PG_WINDOW_WIDTH, self.PG_WINDOW_HEIGHT))
        self.pg_board = []
        self.pg_pieces = []
        self.pg_assets = {}

        for i in range(self.NUM_PIECES):
            col = i % self.PIECE_ATTRIBUTES
            row = i // self.PIECE_ATTRIBUTES

            self.pg_assets[f"{i:04b}"] = pygame.transform.scale(
                    pygame.image.load(f"{settings.games_assets_path}/quarto_assets/{i:04b}.png"),
                    (self.PG_PIECE_WIDTH, self.PG_PIECE_HEIGHT)
                )
            self.pg_assets['-1-1-1-1'] = None

            self.pg_board.append(
                QuartoButton(
                    x=col * (self.PG_PIECE_WIDTH + self.PG_GAP),
                    y=(row + 1) * (self.PG_PIECE_HEIGHT + self.PG_GAP),
                    width=self.PG_PIECE_WIDTH,
                    height=self.PG_PIECE_HEIGHT,
                )
            )

            self.pg_pieces.append(
                QuartoButton(
                    x=(col + 5) * (self.PG_PIECE_WIDTH + self.PG_GAP),
                    y=(row + 1) * (self.PG_PIECE_HEIGHT + self.PG_GAP),
                    width=self.PG_PIECE_WIDTH,
                    height=self.PG_PIECE_HEIGHT,
                )
            )

            self.pg_selected = QuartoButton(
                x=self.PG_WINDOW_WIDTH - self.PG_PIECE_WIDTH,
                y=0,
                width=self.PG_PIECE_WIDTH,
                height=self.PG_PIECE_HEIGHT,
            )

    def _1_vs_randomplayer(self):
        rp = RandomPlayer(action_dim=self.NUM_PIECES * 2)
        running = True
        self.render()
        while running:
            if self.current_player == 0:
                action_dist = rp.forward(x=None, mask=self.get_action_space())
                action = np.argmax(action_dist)
                self.step(action)
                self.render()
                running = not self.is_game_over()
            else:
                action = self._get_human_action()
                self.step(action)
                self.render()
                running = not self.is_game_over()


        self.render()
        match self.score():
            case 0:
                print("Match null")
            case 1:
                print("Random player a gagné")
            case -1:
                print("Vous avez gagné")
    
    def _get_human_action(self) -> int:
        """Bloque jusqu'à ce que le joueur humain clique. Retourne l'action."""
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit

                if self.is_selecting_phase:
                    for i, piece in enumerate(self.pg_pieces):
                        if piece.is_clicked(event):
                            return i
                else:
                    for i, cell in enumerate(self.pg_board):
                        if cell.is_clicked(event) and self.pg_selected.image is not None:
                            return i + self.NUM_PIECES

    def _is_forbidden_action(self, actions):
        action = np.argmax(actions)
        if self.is_selecting_phase:
            p = self._get_piece(action, self.available_pieces)
            if -1 in p:
                return True
        else :
            action -= self.NUM_PIECES
            p = self._get_piece(action, self.board) 
            if 0 in p or 1 in p:
                return True

    def _get_piece(self, pos, tab):
        return tab[pos * self.PIECE_ATTRIBUTES: pos * self.PIECE_ATTRIBUTES + self.PIECE_ATTRIBUTES]
    
    def _update_piece(self, piece, pos, tab):
        tab[pos * self.PIECE_ATTRIBUTES: pos * self.PIECE_ATTRIBUTES + self.PIECE_ATTRIBUTES] = piece

def main():
    env = QuartoEnv()
    env._1_vs_randomplayer()

if __name__ == "__main__":
    main()
