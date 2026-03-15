import pygame
from deeprl_5iabd.config import settings
from deeprl_5iabd.helper import ImageButton
from deeprl_5iabd.envs.base_env import BaseEnv

class LineWorld(BaseEnv):
    """Environnement 1D : ligne de 5 positions (0-4).
    - Position 0 : récompense -1 (terminal)
    - Position 4 : récompense +1 (terminal)
    - Positions 1-3 : récompense 0
    - Actions : 0=gauche, 1=droite
    """

    BOARD_SIZE = 5

    PG_PIECE_W = 250
    PG_PIECE_H = 250
    PG_GAP = 5

    PG_WINDOW_W = (PG_PIECE_W + PG_GAP) * BOARD_SIZE
    PG_WINDOW_H = (PG_PIECE_H + PG_GAP)

    def __init__(self):
        super().__init__("line_world")
        self.agent_pos = 2
        self._pygame_initialized = False

    def reset(self):
        self.agent_pos = 2

    def step(self, action) -> None:
        if self.is_game_over():
            return

        match action:
            case 0: self.agent_pos -= 1
            case 1: self.agent_pos += 1

    def is_game_over(self):
        return  self.agent_pos in [0,4]

    def score(self) -> int:
        if not (0 <= self.agent_pos < 5):
            raise ValueError(f"Error agent_pos {self.agent_pos}: agent hors de la grille")

        match self.agent_pos:
            case 0:
                return -1
            case 4:
                return 1
            case _:
                return 0

    def get_observation_space(self) -> list[int]:
        return [self.agent_pos]
 
    def get_action_space(self) -> list[int]:
        """0: gauche, 1: droite"""
        action = [1,1]
        if self.agent_pos == 0:
            action[0] = 0
        if self.agent_pos == 4:
            action[1] = 0
        return action
    
    def render(self) -> None:
        if not self._pygame_initialized:
            self._init_pygame()

        self.screen.fill((30, 30, 30))

        for i in range(self.BOARD_SIZE):
            self.pg_board[i].image = None
            if i == self.agent_pos:
                self.pg_board[i].image = self.pg_assets[self.last_action]
            self.pg_board[i].draw(self.screen)
        pygame.display.flip()

    def _init_pygame(self):
        pygame.init()
        self.last_action = 1
        self.screen = pygame.display.set_mode((self.PG_WINDOW_W, self.PG_WINDOW_H))
        pygame.display.set_caption("LineWorld")
        self._pygame_initialized = True

        self.pg_assets = [
            pygame.transform.scale(
                pygame.image.load(f"{settings.line_world_assets_path}/{i}.png"),
                (self.PG_PIECE_W, self.PG_PIECE_H)
            )
            for i in range(0, 2)
        ]

        self.pg_board   = [ImageButton(x = c * self.PG_PIECE_W + c * self.PG_GAP, y = 0,
                                        width = self.PG_PIECE_W, height = self.PG_PIECE_H)
                            for c in range(self.BOARD_SIZE)]

    def _play(self):
        self.reset()
        while not self.is_game_over():
            self.render()
            action = None
            while action is None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            action = 0
                            self.last_action = action
                            self.step(action)
                        elif event.key == pygame.K_RIGHT:
                            action = 1
                            self.last_action = action
                            self.step(action)
                        self.render()

if __name__ == "__main__":
    env = LineWorld()
    env._play()