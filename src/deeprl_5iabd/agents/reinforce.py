import torch
from torch import optim
import torch.nn.functional as F
from torch.distributions import Categorical
from deeprl_5iabd.config import settings
from deeprl_5iabd.helper import softmax_with_mask
from deeprl_5iabd.envs.base_env import BaseEnv
from deeprl_5iabd.agents.base_agent import BaseAgent
from deeprl_5iabd.tracking.base_logger import BaseLogger

def reinforce(env: BaseEnv,
              opponent_model: BaseAgent,
              reinforce_agent: BaseAgent,
              logger: BaseLogger | None = None,
              num_episodes: int = 100_000,
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
    for epoch in range(1, num_episodes):
        env.reset()
        states, actions, rewards = [], [], []
        log_probs = []

        while not env.is_game_over():
            if env.player == 0:
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
                probs = opponent_model.forward(x=None, mask=mask)
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

        if logger is not None:
            metrics = {
                "Train/Loss": loss.item(),
                "Train/WinRate": nbr_win / epoch,
                "Train/EpisodeLength": len(states)
            }
            logger.log_dict(metrics, step=epoch)
            
        if epoch % 100 == 0:
            print(f"Manche {epoch} | Gain {nbr_win/epoch:.2f} | Loss {loss.item():.4f}")

    if logger is not None:
        logger.close()

    return reinforce_agent
