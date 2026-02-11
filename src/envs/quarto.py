from envs.base_env import BaseEnv
import pygame

class QuartoEnv():
    # COLOR_IDX = 0;   // light/dark
    # SIZE_IDX = 1;    // large/small
    # SHAPE_IDX = 2;   // square/round
    # FILL_IDX = 3;    // solid/hollow
    
    NUM_PIECES = 16
    PIECE_ATTRIBUTES = 4
    BOARD_SIZE = NUM_PIECES * PIECE_ATTRIBUTES
    
    PG_PIECE_HIGHT = 100
    PG_WINDOW_HIGHT = PG_PIECE_HIGHT * (PIECE_ATTRIBUTES * 2) + PG_PIECE_HIGHT
    PG_WINDOW_WIDTH = PG_PIECE_HIGHT * PIECE_ATTRIBUTES + PG_PIECE_HIGHT
    
    VICTORY_PATTERNS = [
        [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], # Lignes horizontales
        [0, 4, 8, 12],   [1, 5, 9, 13],   [2, 6, 10, 14],  [3, 7, 11, 15], # Colonnes verticales
        [0, 5, 10, 15], [3, 6, 9, 12] # Diagonales
    ]

    def __init__(self):
        # super().__init__("quarto")
        pygame.init()
        self.scrn = pygame.display.set_mode((self.PG_WINDOW_HIGHT, self.PG_WINDOW_WIDTH))

        self.pg_piece_assets = [pygame.image.load("/home/adam/Documents/esgi/drl/deepRL_5a/myPythonLib/game_assets/quarto_assets/0000.png") for _ in range(self.NUM_PIECES)]

        self.reset()

    def render_pygame(self):
        def display_game():
            self.scrn.blit(self.pg_piece_assets[0], (0, 0))
            pygame.display.flip()

        who = 1
        self.reset()
        while 1:
            display_game()
            # self._render_terminal()
            # p_idx = input(f"player {who} select a piece ? ...")
            # piece = self._select_piece(int(p_idx))
            # who = int(not who)
            # print(f"Selected piece for {who}: {piece}")
            # pos = input(f"player {who} select pos number ? ...")
            # self._add_piece(piece, int(pos))

        self._render_terminal()
        print(f"Player {who} Won !")

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
        if -1 not in self.available_pieces:
            return True

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
