import pygame
import numpy as np
from deeprl_5iabd.envs.model_based_env import ModelBasedEnv
from deeprl_5iabd.config import settings
from deeprl_5iabd.helper import ImageButton
from deeprl_5iabd.agents.random_agent import RandomPlayer

class GridWorld(ModelBasedEnv):
    BOARD_SIZE = 5

    PG_PIECE_W = 150
    PG_PIECE_H = 150
    PG_GAP = 5

    PG_WINDOW_W = (PG_PIECE_W + PG_GAP) * BOARD_SIZE
    PG_WINDOW_H = (PG_PIECE_H + PG_GAP) * BOARD_SIZE

    def __init__(self):
        super().__init__("GridWorld")
        self.reset()
        self.T = [(0,4), (4,4)]
        self.A = [0, 1, 2, 3]     # 0=bas, 1=haut, 2=droite, 3=gauche
        self.R = [-3.0, 0.0, 1.0]
        self.p_matrix = self._create_p_matrix()
        self._pygame_initialized = False

    def num_states(self):
        return 25

    def num_actions(self):
        return 4

    def num_rewards(self):
        return 3

    def state_id(self, state) -> int:
        return state[0] * self.BOARD_SIZE + state[1]

    def available_actions(self) -> np.ndarray:
        if self.is_game_over():
            return np.array([])
        am = self.get_action_space()
        return np.array([a for a in self.A if am[a] == 1])

    def reset(self) -> None:
        self.board = np.full((self.BOARD_SIZE, self.BOARD_SIZE), -1)
        self.agent_pos = (0, 0)
        self.board[self.agent_pos] = 1

    def is_game_over(self) -> bool:
        return self.agent_pos == (4, 4) or self.agent_pos == (0, 4)

    def get_observation_space(self):
        return list(self.agent_pos)

    def get_action_space(self) -> list[int]:
        picks = np.ones(4)
        if self.agent_pos[0] == 0:
            picks[1] = 0
        if self.agent_pos[1] == 0:
            picks[3] = 0
        if self.agent_pos[0] == self.BOARD_SIZE - 1:
            picks[0] = 0
        if self.agent_pos[1] == self.BOARD_SIZE - 1:
            picks[2] = 0
        return picks.tolist()

    def step(self, action: int) -> None:
        """0: bas, 1: haut, 2: droite, 3: gauche"""
        if self.get_action_space()[action] == 0:
            return

        self.board[self.agent_pos] = -1
        if action == 0:
            self.agent_pos = (self.agent_pos[0] + 1, self.agent_pos[1])
        elif action == 1:
            self.agent_pos = (self.agent_pos[0] - 1, self.agent_pos[1])
        elif action == 2:
            self.agent_pos = (self.agent_pos[0], self.agent_pos[1] + 1)
        elif action == 3:
            self.agent_pos = (self.agent_pos[0], self.agent_pos[1] - 1)
        self.board[self.agent_pos] = 1

    def score(self) -> int:
        if self.agent_pos == (4, 4):
            return self.R[0]
        elif self.agent_pos == (0, 4):
            return self.R[2]
        else:
            return self.R[1]

    def render(self) -> None:
        if not self._pygame_initialized:
            self._init_pygame()
            self._pygame_initialized = True
        
        self.screen.fill((30, 30, 30))
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if self.board[r, c] == 1:
                    self.pg_board[r][c].image = self.pg_assets[self.last_action]
                else:
                    self.pg_board[r][c].image = None
                self.pg_board[r][c].draw(self.screen)
        pygame.display.flip()

    def _create_p_matrix(self):
        # 0=bas, 1=haut, 2=droite, 3=gauche
        # 0=-3.0,  1=0.0,    2=1.0
        # p[s, a, s_prime, r_idx]
        p = np.zeros((25, 4, 25, 3))
        terminal_ids = [4, 24]

        down_border = [20, 21, 22, 23, 24]
        up_border = [0, 1, 2, 3, 4]
        left_border = [0, 5, 10, 15, 20]
        right_border = [4, 9, 14, 19, 24]

        for s in range(self.num_states()):
            if s in terminal_ids:
                continue

            for a in range(4):
                is_collision = False
                next_s = s

                if a == 0:
                    is_collision = s in down_border
                    next_s = s + 5
                elif a == 1:
                    is_collision = s in up_border
                    next_s = s - 5
                elif a == 2:
                    is_collision = s in right_border
                    next_s = s + 1
                elif a == 3:
                    is_collision = s in left_border
                    next_s = s - 1

                # la récompense
                if is_collision:
                    p[s, a, s, 1] = 1.0
                else:
                    r_idx = 1 # 0 par défaut
                    if next_s == 4: r_idx = 2  # 1 (Goal)
                    elif next_s == 24: r_idx = 0 # -3 (Trap)

                    p[s, a, next_s, r_idx] = 1.0

        return p

    def _init_pygame(self) -> None:
        pygame.init()
        self.last_action = 0
        self.screen = pygame.display.set_mode((self.PG_WINDOW_W, self.PG_WINDOW_H))
        pygame.display.set_caption("GridWorld")

        self.pg_assets = [
            pygame.transform.scale(
                pygame.image.load(
                    f"{settings.grid_world_assets_path}/{i}.png"
                ),
                (self.PG_PIECE_W, self.PG_PIECE_H),
            )
            for i in range(0, 4)
        ]
        self.pg_board = [
            [
                ImageButton(
                    c * (self.PG_PIECE_W + self.PG_GAP),
                    r * (self.PG_PIECE_H + self.PG_GAP),
                    self.PG_PIECE_W,
                    self.PG_PIECE_H,
                )
                for c in range(self.BOARD_SIZE)
            ]
            for r in range(self.BOARD_SIZE)
        ]

  # Game modes
    def _play(self):
        self.reset()
        while not self.is_game_over():
            self.render()
            action = None
            actions = self.get_action_space()
            while action is None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            action = 3
                        elif event.key == pygame.K_RIGHT:
                            action = 2
                        elif event.key == pygame.K_UP:
                            action = 1
                        elif event.key == pygame.K_DOWN:
                            action = 0

                        if actions[action] == 1:
                            self.last_action = action
                            self.step(action)
                            self.render()

if __name__ == "__main__":
    env = GridWorld()
    env._play()
