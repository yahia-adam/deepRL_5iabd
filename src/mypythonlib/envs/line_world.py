from unittest import case

from envs.base_env import MonteCarloEnv

class LineWorld(MonteCarloEnv):
    """Environnement 1D : ligne de 5 positions (0-4).
    - Position 0 : récompense -1 (terminal)
    - Position 4 : récompense +1 (terminal)
    - Positions 1-3 : récompense 0
    - Actions : 0=gauche, 1=droite
    """

    def __init__(self):
        super().__init__()
        self.agent_pos = 2

    def state(self) -> int:
        return self.agent_pos

    def num_states(self) -> int:
        return 5

    def num_actions(self) -> int:
        return 2

    def step(self, action):
        if self.is_game_over():
            return

        match action:
            case 0: self.agent_pos -= 1
            case 1: self.agent_pos += 1

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

    def is_game_over(self):
        return  self.agent_pos in [0,4]

    def reset(self):
        self.agent_pos = 2
