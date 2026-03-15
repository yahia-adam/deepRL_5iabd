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
              early_stop = False,
              early_stop_val = 0.6,
        ):

    optimizer = optim.Adam(reinforce_agent.parameters(), lr=lr)

    nbr_win, nbr_loss, nbr_draw = 0, 0, 0
    for epoch in range(1, num_episodes):
        log_probs, rewards = [], []

        env.reset()
        while not env.is_game_over():
            if env.player == 0:
                mask = env.get_action_space()
                np_state = env.get_observation_space()

                tensor_state = torch.tensor(np_state).float()
                probs = reinforce_agent.forward(tensor_state, mask)
                probs_dist = Categorical(probs)
                action_pos = probs_dist.sample()

                log_probs.append(probs_dist.log_prob(action_pos))

                env.step(action_pos.item())
                rewards.append(env.score())
            else:
                mask = env.get_action_space()
                probs = opponent_model.forward(x=None, mask=mask)
                probs_dist = Categorical(probs)
                action_pos = probs_dist.sample()
                env.step(action_pos.item())

        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss += -log_prob * G

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
                "Train/EpisodeLength": len(rewards)
            }
            logger.log_dict(metrics, step=epoch)
            
        if epoch % 100 == 0:
            print(f"Manche {epoch} | Gain {nbr_win/epoch:.2f} | Loss {loss.item():.4f}")

    if logger is not None:
        logger.close()

    return reinforce_agent
