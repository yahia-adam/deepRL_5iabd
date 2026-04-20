import time
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GridWorldEnv(gym.Env):
    """
    Environnement Grille 2D (5x5).
    - Position initiale : (0, 0)
    - Position (4, 4) : récompense +1.0 (terminal)
    - Position (0, 4) : récompense -1.0 (terminal)
    - Actions : 0=bas, 1=haut, 2=droite, 3=gauche
    """
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super().__init__()

        self.size = 5
        self.render_mode = render_mode

        # Espace d'observation : tableau [ligne, colonne] allant de [0, 0] à [4, 4]
        self.observation_space = spaces.MultiDiscrete([self.size, self.size])

        # Espace d'action : 4 actions possibles
        self.action_space = spaces.Discrete(4) 

        # Variables Pygame
        self.window_size = 500  # 500x500 pixels (100 pixels par case)
        self.cell_size = self.window_size // self.size
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # L'agent commence toujours en haut à gauche (ligne 0, colonne 0)
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        return self.agent_pos, {}

    def step(self, action):
        row, col = self.agent_pos

        # 1. Appliquer l'action (avec blocage aux bords)
        if action == 0:   # Bas
            row = min(self.size - 1, row + 1)
        elif action == 1: # Haut
            row = max(0, row - 1)
        elif action == 2: # Droite
            col = min(self.size - 1, col + 1)
        elif action == 3: # Gauche
            col = max(0, col - 1)

        self.agent_pos = np.array([row, col], dtype=np.int32)

        # 2. Vérifier si l'état est terminal et calculer la récompense
        terminated = False
        reward = 0.0

        if row == 4 and col == 4:
            terminated = True
            reward = 1.0
        elif row == 0 and col == 4:
            terminated = True
            reward = -1.0

        return self.agent_pos, reward, terminated, False, {}

    def render(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("GridWorld Gym")
            self.clock = pygame.time.Clock()

        self.window.fill((30, 30, 30)) # Fond sombre

        for r in range(self.size):
            for c in range(self.size):
                # Attention dans Pygame : X=colonne, Y=ligne
                rect = (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                
                # Couleurs par défaut (Gris)
                color = (100, 100, 100)
                
                # Coloration des cases spéciales
                if r == 4 and c == 4: color = (50, 200, 50)   # Vert : Récompense +1
                if r == 0 and c == 4: color = (200, 50, 50)   # Rouge : Récompense -1
                
                pygame.draw.rect(self.window, color, rect)
                pygame.draw.rect(self.window, (0, 0, 0), rect, 2) # Bordure

        # Dessiner l'agent (Cercle bleu)
        agent_r, agent_c = self.agent_pos
        center_x = agent_c * self.cell_size + self.cell_size // 2
        center_y = agent_r * self.cell_size + self.cell_size // 2
        pygame.draw.circle(self.window, (50, 150, 255), (center_x, center_y), self.cell_size // 3)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None


if __name__ == "__main__":
    env = GridWorldEnv(render_mode="human")
    obs, info = env.reset()
    env.render()

    print("Jouez avec les flèches directionnelles. Atteignez la case verte ! Fermez la fenêtre pour quitter.")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                action = None
                if event.key == pygame.K_DOWN:  action = 0
                if event.key == pygame.K_UP:    action = 1
                if event.key == pygame.K_RIGHT: action = 2
                if event.key == pygame.K_LEFT:  action = 3

                if action is not None:
                    obs, reward, terminated, truncated, info = env.step(action)
                    env.render()

                    if terminated:
                        print(f"Fin de partie ! Récompense obtenue : {reward}")
                        env.reset()
                        env.render()

    env.close()