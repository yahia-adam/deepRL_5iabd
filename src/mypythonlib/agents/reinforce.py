import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from mypythonlib.config import settings
from mypythonlib.envs.base_env import BaseEnv

class ReinforceModel(nn.Module):
    def __init__(self, input_size, output_size, device = settings.device):
        super().__init__()
        torch.set_default_device(device)
        self.model = nn.Sequential(
            nn.Linear(764, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x

def reinforce(env: BaseEnv, nbr_episode: int = 10000, lr: float = 0.001, Rinforce_agent = ReinforceModel()):

    loss = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(Rinforce_agent.model.parameters(), lr=lr)
    for _ in range(nbr_episode):
        env.reset()
        states, actions, rewards = [], [], []
        while not env.is_game_over():
            s = env.get_observation_space()
            a = Rinforce_agent.model(s)
            states.append(s)
            actions.append(a)
            rewards.append(0)

            env.step(a)

        r = env.score()


    return Rinforce_agent