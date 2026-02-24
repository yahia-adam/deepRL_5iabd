import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.distributions import Categorical
from mypythonlib.config import settings
from mypythonlib.helper import RandomPlayer, softmax_with_mask
from mypythonlib.envs.base_env import BaseEnv
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReinforceModel(nn.Module):
    def __init__(self, input_size, output_size, layers=[512, 256, 128]):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Première couche
        self.layers.append(nn.Linear(input_size, layers[0]))
        
        # Couches cachées avec activations
        for i in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[i-1], layers[i]))
        
        # Couche de sortie (Logits bruts)
        self.output_layer = nn.Linear(layers[-1], output_size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = F.relu(layer(x))
        
        logits = self.output_layer(x)
        return softmax_with_mask(logits, mask)

def reinforce(env: BaseEnv, nbr_episode: int = 1_000_000_000, lr: float = 0.001, gamma: float = 0.99):
    reinforce_agent = ReinforceModel(input_size=len(env.get_observation_space()), output_size=len(env.get_action_space()))
    optimizer = torch.optim.Adam(reinforce_agent.parameters(), lr=lr)
    rp = RandomPlayer(len(env.get_action_space()))

    nbr_game, nbr_win, nbr_loss, nbr_draw = 0, 0, 0, 0
    for _ in range(nbr_episode):
        env.reset()
        states, actions, rewards = [], [], []
        log_probs = []

        while not env.is_game_over():
            if env.current_player == 0:
                mask = env.get_action_space()
                np_state = env.get_observation_space()

                tensor_state = torch.tensor(np_state).float()
                probs = reinforce_agent.forward(tensor_state, mask)
                probs_dist = Categorical(probs)
                action_pos = probs_dist.sample()

                states.append(np_state)
                actions.append(action_pos.item())
                log_probs.append(probs_dist.log_prob(action_pos))

                env.step(action_pos.item())
                rewards.append(env.score()) 
            else:
                mask = env.get_action_space()
                probs = rp.play(mask)
                probs_dist = Categorical(probs)
                action_pos = probs_dist.sample()
                env.step(action_pos)

        # 2. MISE À JOUR DES PARAMÈTRES THETA (À LA FIN DE L'ÉPISODE)
        loss = 0
        for t in range(len(states)):
            # Calcul du gain G := somme des gamma^k * R_k
            G = 0
            pw = 0
            for k in range(t, len(rewards)):
                G += (gamma ** pw) * rewards[k]
                pw += 1
            
            # Formule de mise à jour : theta + alpha * gamma^t * G * grad ln(pi)
            # En PyTorch, on fait une descente de gradient, donc on minimise l'opposé (-G)
            loss += - (gamma ** t) * G * log_probs[t]

        # Optimisation
        optimizer.zero_grad()
        loss.backward() # Calcule le gradient total
        optimizer.step() # Applique la mise à jour aux paramètres theta

        nbr_game += 1
        score = env.score()
        if score == 1:
            nbr_win += 1
        elif score == -1:
            nbr_loss += 1
        else:
            nbr_draw += 1
        
        if (nbr_game % 100 == 0):
            print(f"%de gain {nbr_win / nbr_game} gagné: {nbr_win} perdu: {nbr_loss} egalité: {nbr_draw} manch: {nbr_game}")
    return reinforce_agent