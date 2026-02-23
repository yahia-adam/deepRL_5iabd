import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from mypythonlib.config import settings
from mypythonlib.helper import RandomPlayer, softmax_with_mask
from mypythonlib.envs.base_env import BaseEnv

class ReinforceModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x, mask):
        x = self.model(x)
        return softmax_with_mask(x, mask)

def reinforce(env: BaseEnv, nbr_episode: int = 10000, lr: float = 0.001):

    Rinforce_agent = ReinforceModel(input_size=len(env.get_action_space()), output_size=len(env.get_action_space()))
    optimizer = torch.optim.Adam(Rinforce_agent.model.parameters(), lr=lr)
    rp = RandomPlayer(len(env.get_action_space()))
    for _ in range(nbr_episode):
        env.reset()
        states, actions, rewards = [], [], []
        while not env.is_game_over():
            if env.current_player == 0:
                s = env.get_observation_space()
                a = Rinforce_agent.model(s)
                states.append(s)
                actions.append(a)
                rewards.append(env.score())
            else :
                env.step(rp.play())

    return Rinforce_agent
