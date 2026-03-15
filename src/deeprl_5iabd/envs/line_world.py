from deeprl_5iabd.envs.base_env import BaseEnv
import pygame

class LineWorld(BaseEnv):
    """Environnement 1D : ligne de 5 positions (0-4).
    - Position 0 : récompense -1 (terminal)
    - Position 4 : récompense +1 (terminal)
    - Positions 1-3 : récompense 0
    - Actions : 0=gauche, 1=droite
    """

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
        action = [1,1]
        if self.agent_pos == 0:
            action[0] = 0
        if self.agent_pos == 4:
            action[1] = 0
        return action
    
    def render(self) -> None:
        if not self._pygame_initialized:
            pygame.init()
            self.scrn = pygame.display.set_mode((500, 100))
            pygame.display.set_caption("LineWorld")
            self._pygame_initialized = True

        self.scrn.fill((30, 30, 30))

        for i in range(5):
            color = (80, 200, 80) if i == self.agent_pos else (80, 80, 80)
            pygame.draw.rect(self.scrn, color, (i * 100, 10, 80, 80))

        pygame.display.flip()

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
                        elif event.key == pygame.K_RIGHT:
                            action = 1
            self.step(action)
        self.render()

if __name__ == "__main__":
    env = LineWorld()
    env._play()