import pygame
from mypythonlib.config import settings
from mypythonlib.envs.base_env import BaseEnv

class Button:
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

class QuartoEnv():
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
        # super().__init__("quarto")
        self.reset()
        self.init_pygame()

    def init_pygame(self):
        pygame.init()
        self.scrn = pygame.display.set_mode((self.PG_WINDOW_WIDTH, self.PG_WINDOW_HEIGHT))
        self.pg_board = []
        self.pg_pieces = []
        self.current_player = 0

        for i in range(self.NUM_PIECES):
            col = i % self.PIECE_ATTRIBUTES
            row = i // self.PIECE_ATTRIBUTES

            asset = pygame.transform.scale(
                pygame.image.load(f"game_assets/quarto_assets/{i:04b}.png"),
                (self.PG_PIECE_WIDTH, self.PG_PIECE_HEIGHT)
            )

            self.pg_board.append(
                Button(
                    x=col * (self.PG_PIECE_WIDTH + self.PG_GAP),
                    y=(row + 1) * (self.PG_PIECE_HEIGHT + self.PG_GAP),
                    width=self.PG_PIECE_WIDTH,
                    height=self.PG_PIECE_HEIGHT,
                )
            )

            self.pg_pieces.append(
                Button(
                    x=(col + 5) * (self.PG_PIECE_WIDTH + self.PG_GAP),
                    y=(row + 1) * (self.PG_PIECE_HEIGHT + self.PG_GAP),
                    width=self.PG_PIECE_WIDTH,
                    height=self.PG_PIECE_HEIGHT,
                    image=asset
                )
            )

            self.pg_selected = Button(
                x=self.PG_WINDOW_WIDTH - self.PG_PIECE_WIDTH,
                y=0,
                width=self.PG_PIECE_WIDTH,
                height=self.PG_PIECE_HEIGHT,
            )

    def render_pygame(self):

        def get_status_text():
            if self.pg_selected.image is None:
                return f"Player {self.current_player + 1} sélectionne une pièce"
            else:
                return f"Player {self.current_player + 1} place la pièce"

        def render_header():
            font = pygame.font.SysFont(None, 36)
            surface = font.render(get_status_text(), True, (255, 255, 255))
            self.scrn.blit(surface, (10, self.PG_PIECE_HEIGHT // 2 - surface.get_height() // 2))

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                for i, piece in enumerate(self.pg_pieces):
                    if piece.is_clicked(event):
                        self.pg_selected.image = piece.image
                        self.pg_pieces[i].image = None

                for i, cell in enumerate(self.pg_board):
                    if cell.is_clicked(event) and self.pg_selected.image is not None:
                        self.pg_board[i].image = self.pg_selected.image
                        self.pg_selected.image = None

            # Rendu
            self.scrn.fill((0, 0, 0))
            render_header()
            for b in self.pg_board:
                b.draw(self.scrn)
            for p in self.pg_pieces:
                p.draw(self.scrn)
            self.pg_selected.draw(self.scrn)

            pygame.display.flip()

            if self.is_game_over():
                running = False

        pygame.quit()

    def reset(self):
        self.all_pieces = [
            1, 1, 1, 1,  1, 0, 0, 0,  1, 0, 0, 1,  1, 1, 0, 0,
            1, 1, 0, 1,  1, 0, 1, 0,  1, 0, 1, 1,  1, 1, 1, 0,
            0, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 1,  0, 1, 0, 0,
            0, 1, 0, 1,  0, 0, 1, 0,  0, 0, 1, 1,  0, 1, 1, 0,
        ]
        self.available_pieces = self.all_pieces.copy()
        self.board = [-1 for _ in range(16 * 4)]

    def is_game_over(self):
        for e in self.VICTORY_PATTERNS:
            for i in range(self.PIECE_ATTRIBUTES):
                cel_0 = self.board[e[0] * self.PIECE_ATTRIBUTES + i]
                cel_1 = self.board[e[1] * self.PIECE_ATTRIBUTES + i]
                cel_2 = self.board[e[2] * self.PIECE_ATTRIBUTES + i]
                cel_3 = self.board[e[3] * self.PIECE_ATTRIBUTES + i]
                if cel_0 == cel_1 and cel_1 == cel_2 and cel_2 == cel_3 and cel_3 != -1:
                    return True
        
        # egalité
        # if -1 not in self.available_pieces:
        #     return True

        return False

    def _render_terminal(self):
        print("available pieces:")
        count = 0
        for i in range(0, self.NUM_PIECES * self.PIECE_ATTRIBUTES, self.PIECE_ATTRIBUTES):
            print(f"{count} : {self.available_pieces[i: i + self.PIECE_ATTRIBUTES]}")
            count += 1
        
        print("\nGame borad:")
        for i in range(0, self.BOARD_SIZE, self.PIECE_ATTRIBUTES):
            if (i % (self.PIECE_ATTRIBUTES * self.PIECE_ATTRIBUTES) == 0 and i != 0):
                print("")
            print(f"{self.board[i: i + self.PIECE_ATTRIBUTES]}", end=" ")
        print("\n")

    def _select_piece(self, pos):
        p = self.available_pieces[pos * self.PIECE_ATTRIBUTES: pos * self.PIECE_ATTRIBUTES + self.PIECE_ATTRIBUTES]
        self.available_pieces[pos * self.PIECE_ATTRIBUTES: pos * self.PIECE_ATTRIBUTES + self.PIECE_ATTRIBUTES] = [-1 for _ in range(self.PIECE_ATTRIBUTES)]
        return p
    
    def _add_piece(self, piece, pos):
        for i,e in enumerate(piece):
            self.board[pos*self.PIECE_ATTRIBUTES + i] = e

    def _2v2_game(self):
        who = 1
        self.reset()

        while not self.is_game_over():
            self._render_terminal()
            p_idx = input(f"player {who} select a piece ? ...")
            piece = self._select_piece(int(p_idx))
            who = int(not who)
            print(f"Selected piece for {who}: {piece}")
            pos = input(f"player {who} select pos number ? ...")
            self._add_piece(piece, int(pos))

        self._render_terminal()
        print(f"Player {who} Won !")

def main():
    env = QuartoEnv()
    env.render_pygame()
    
if __name__ == "__main__":
    main()
