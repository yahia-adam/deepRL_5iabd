import torch
from torch import optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from mypythonlib.config import settings
from mypythonlib.helper import softmax_with_mask
from mypythonlib.envs.base_env import BaseEnv
from mypythonlib.agents.agent_base import MyModel
from mypythonlib.agents.random_agent import RandomPlayer

def reinforce(env: BaseEnv,
              oponent_model: MyModel,
              reinforce_agent: MyModel,
              nbr_episode: int = 100_000,
              lr: float = 0.001,
              gamma: float = 0.99,
              log_dir: str = settings.training_logs_path,
              model_dir: str = settings.models_path,
              model_name: str = "reinforce",
              early_stop = False,
              early_stop_val = 0.6,
        ):

    optimizer = optim.Adam(reinforce_agent.parameters(), lr=lr)
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
                probs = oponent_model.forward(x=None, mask=mask)
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
