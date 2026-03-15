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
              num_episodes: int = 5_000,
              lr: float = 0.001,
              gamma: float = 0.99,
              early_stop = False,
              early_stop_val = 0.65,
        ):

    optimizer = optim.Adam(reinforce_agent.parameters(), lr=lr)

    win_history = [] 
    window_size = 1_000

    for epoch in range(1, num_episodes + 1):
        log_probs, rewards = [], []

        env.reset()
        while not env.is_game_over():
            mask = env.get_action_space()
            np_state = env.get_observation_space()
            tensor_state = torch.tensor(np_state).float()
            if env.player == 0:
                probs = reinforce_agent.forward(tensor_state, mask)
                probs_dist = Categorical(probs)
                action_pos = probs_dist.sample()
                env.step(action_pos.item())

                log_probs.append(probs_dist.log_prob(action_pos))
                rewards.append(env.score())
            else:
                with torch.no_grad():
                    probs = opponent_model.forward(x=tensor_state, mask=mask)
                probs_dist = Categorical(probs)
                action_pos = probs_dist.sample()
                env.step(action_pos.item())

        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns).float()
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss += -log_prob * G

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        score = env.score()
        current_win = 1 if score == 1 else 0
        win_history.append(current_win)
        if len(win_history) > window_size:
            win_history.pop(0)
        rolling_win_rate = sum(win_history) / len(win_history)

        if logger is not None:
            metrics = {
                "Train/Loss": loss.item(),
                "Train/WinRate_Rolling": rolling_win_rate,
                "Train/EpisodeLength": len(rewards)
            }
            logger.log_dict(metrics, step=epoch)
        
        if early_stop and len(win_history) >= window_size:
            if rolling_win_rate >= early_stop_val:
                print(f"\n[EARLY STOP] Objectif atteint à l'époque {epoch} !")
                print(f"Win Rate sur les {window_size} derniers matchs : {rolling_win_rate:.2f}")
                break

        if epoch % window_size == 0:
            print(f"Manche {epoch} | Win Rate {rolling_win_rate} | Loss {loss.item():.4f}")

    if logger is not None:
        logger.close()

    return reinforce_agent
