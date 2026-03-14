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
from torch.utils.tensorboard import SummaryWriter

class ReinforceModel(nn.Module):
    def __init__(self, input_size, output_size, hiddenlayers=[512, 256, 128]):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hiddenlayers[0]))
        for i in range(1, len(hiddenlayers)):
            self.layers.append(nn.Linear(hiddenlayers[i-1], hiddenlayers[i]))
        self.output_layer = nn.Linear(hiddenlayers[-1], output_size)

        self.config = {"input_size": input_size, "output_size": output_size, "layers": hiddenlayers}

    def forward(self, x, mask):
        for layer in self.layers:
            x = F.relu(layer(x))

        logits = self.output_layer(x)
        return softmax_with_mask(logits, mask)

    def save(self, filename="reinforce_model.pth"):
        path = settings.models_path / filename
        checkpoint = {
            "state_dict": self.state_dict(),
            "config": self.config
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, filename="reinforce_model.pth"):
        path = settings.models_path / filename
        checkpoint = torch.load(path, weights_only=False)
        
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

def reinforce(env: BaseEnv,
              nbr_episode: int = 100_000,
              lr: float = 0.001,
              gamma: float = 0.99,
              log_dir: str = settings.training_logs_path,
              model_dir: str = settings.models_path,
              model_name: str = "reinforce",
              early_stop = False,
              early_stop_val = 0.6,
              oponent_model = None,
        ):

    reinforce_agent = ReinforceModel(input_size=len(env.get_observation_space()), output_size=len(env.get_action_space()))
    optimizer = torch.optim.Adam(reinforce_agent.parameters(), lr=lr)
    if not oponent_model:
        oponent_model = RandomPlayer(len(env.get_action_space()))
    # writer = SummaryWriter()

    nbr_win, nbr_loss, nbr_draw = 0, 0, 0
    for epoch in range(1, nbr_episode):
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
                probs = oponent_model.play(mask)
                probs_dist = Categorical(probs)
                action_pos = probs_dist.sample()
                env.step(action_pos.item())

        loss = 0
        for t in range(len(states)):
            # G := somme des gamma^k * R_k
            G = 0
            pw = 0
            for k in range(t, len(rewards)):
                G += (gamma ** pw) * rewards[k]
                pw += 1
            
            # loss = theta + alpha * gamma^t * G * grad ln(pi)
            loss += - (gamma ** t) * G * log_probs[t]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        score = env.score()
        if score == 1:
            nbr_win += 1
        elif score == -1:
            nbr_loss += 1
        else:
            nbr_draw += 1

        if (epoch % 100 == 0):
            print(f"%de gain {nbr_win / epoch} gagné: {nbr_win} perdu: {nbr_loss} egalité: {nbr_draw} manch: {epoch}")
            # writer.add_scalar(
            #     f"{log_dir}/{model_name}", 
            #     {"pourcentage de gain": (nbr_win / epoch)})
        epoch += 1
    
    # writer.flush()
    return reinforce_agent
