import time
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class LineWorldEnv(gym.Env):
    """
    Environnement 1D : ligne de 5 positions (0-4).
    - Position 0 : récompense -1 (terminal)
    - Position 4 : récompense +1 (terminal)
    - Positions 1-3 : récompense 0
    - Actions : 0=gauche, 1=droite
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        
        self.size = 5
        self.render_mode = render_mode

        # Architecture Gym : Définition des espaces
        self.observation_space = spaces.Discrete(self.size)
        self.action_space = spaces.Discrete(2) # 0: Gauche, 1: Droite

        # Variables Pygame
        self.window_size = 1000
        self.cell_size = self.window_size // self.size
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        # Obligatoire dans Gym pour la reproductibilité
        super().reset(seed=seed)

        self.agent_pos = 2 # Position initiale
        return self.agent_pos, {} # Retourne (observation, info)

    def step(self, action):
        # 1. Appliquer l'action
        if action == 0:   # Gauche
            self.agent_pos = max(0, self.agent_pos - 1)
        elif action == 1: # Droite
            self.agent_pos = min(self.size - 1, self.agent_pos + 1)

        # 2. Vérifier si l'état est terminal
        terminated = (self.agent_pos == 0) or (self.agent_pos == 4)

        # 3. Calculer la récompense
        if self.agent_pos == 0:
            reward = -1.0
        elif self.agent_pos == 4:
            reward = 1.0
        else:
            reward = 0.0

        # Retourne : observation, reward, terminated, truncated, info
        return self.agent_pos, reward, terminated, False, {}

    def render(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.cell_size))
            pygame.display.set_caption("LineWorld Gym")
            self.clock = pygame.time.Clock()

        self.window.fill((30, 30, 30)) # Fond sombre

        for i in range(self.size):
            rect = (i * self.cell_size, 0, self.cell_size, self.cell_size)
            
            # Couleurs des cases (Rouge = -1, Vert = +1, Gris = neutre)
            color = (100, 100, 100)
            if i == 0: color = (200, 50, 50)
            if i == 4: color = (50, 200, 50)
            
            pygame.draw.rect(self.window, color, rect)
            pygame.draw.rect(self.window, (0, 0, 0), rect, 2) # Bordure

            # Dessiner l'agent (Cercle bleu)
            if i == self.agent_pos:
                center = (i * self.cell_size + self.cell_size // 2, self.cell_size // 2)
                pygame.draw.circle(self.window, (50, 150, 255), center, self.cell_size // 3)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None

if __name__ == "__main__":
    env = LineWorldEnv(render_mode="human")
    obs, info = env.reset()
    env.render()

    print("Jouez avec les flèches GAUCHE et DROITE. Fermez la fenêtre pour quitter.")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                action = None
                if event.key == pygame.K_LEFT:  action = 0
                if event.key == pygame.K_RIGHT: action = 1

                if action is not None:
                    obs, reward, terminated, truncated, info = env.step(action)
                    env.render()

                    if terminated:
                        print(f"Fin de partie ! Récompense obtenue : {reward}")
                        env.reset()
                        env.render()

    env.close()