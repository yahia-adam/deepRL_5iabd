import os
import torch
import pickle
import numpy as np
import torch.nn as nn
import gymnasium as gym
from torch import optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

from deeprl_5iabd.helper import softmax_with_mask, plot_rl_dashboard
from deeprl_5iabd.envs.line_world import LineWorldEnv
from deeprl_5iabd.envs.grid_world import GridWorldEnv
from deeprl_5iabd.envs.tictactoe import TicTacToeEnv
from deeprl_5iabd.envs.quarto import QuartoEnv, Phase
from deeprl_5iabd.config import settings


# =========================================================
# Policy Network
# =========================================================
class ReinforceAgent(nn.Module):
    def __init__(self, env):
        super().__init__()

        obs_size = np.array(env.observation_space.shape).prod()
        action_size = env.action_space.n

        self.network = nn.Sequential(
            nn.Linear(obs_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_size),
        )

    def forward(self, state, action_mask):
        logits = self.network(state)
        probs = softmax_with_mask(logits, action_mask)
        return probs


# =========================================================
# Compute discounted returns (Monte Carlo)
# =========================================================
def compute_returns(rewards, gamma):
    returns = []

    for t in range(len(rewards)):
        G_t = 0
        power = 0

        for r in rewards[t:]:
            G_t += (gamma ** power) * r
            power += 1

        returns.append(G_t)

    return returns


# =========================================================
# Policy gradient loss (REINFORCE)
# =========================================================
def compute_policy_loss(log_probs, returns):
    loss = 0

    for log_prob_t, return_t in zip(log_probs, returns):
        loss += -log_prob_t * return_t

    return loss


# =========================================================
# Training loop
# =========================================================
def reinforce(
    env: gym.Env,
    reinforce_agent: ReinforceAgent = None,
    num_episodes: int = 10_000,
    lr: float = 0.001,
    gamma: float = 0.99,
    with_baseline: bool = True,  # (not used here)
):

    # -----------------------------------------
    # Init agent
    # -----------------------------------------
    if reinforce_agent is None:
        reinforce_agent = ReinforceAgent(env)

    optimizer = optim.Adam(reinforce_agent.parameters(), lr=lr)

    rewards_history = np.zeros(num_episodes)
    loss_history = np.zeros(num_episodes)

    # =========================================================
    # Training loop over episodes
    # =========================================================
    for episode in range(num_episodes):

        log_probs_episode = []
        rewards_episode = []

        state, _ = env.reset()
        done = False

        # =====================================================
        # Interaction loop
        # =====================================================
        while not done:

            action_mask = env.get_action_mask()

            # =================================================
            # SPECIAL CASE: Quarto environment
            # =================================================
            if isinstance(env, QuartoEnv):

                # -------------------------
                # PHASE: PLACE
                # -------------------------
                if env.current_player == env.agent_player:
                    if env.phase == Phase.PLACE:

                        state_tensor = torch.tensor(state).float()

                        action_probs = reinforce_agent(state_tensor, action_mask)
                        dist = Categorical(action_probs)

                        action = dist.sample()
                        log_probs_episode.append(dist.log_prob(action))

                        state, reward, done, truncated, _ = env.step(action.item())
                        rewards_episode.append(reward)

                        if done or truncated:
                            break

                    # -------------------------
                    # PHASE: SELECT
                    # -------------------------
                    if env.phase == Phase.SELECT:

                        state_tensor = torch.tensor(state).float()

                        action_probs = reinforce_agent(state_tensor, action_mask)
                        dist = Categorical(action_probs)

                        action = dist.sample()
                        log_probs_episode.append(dist.log_prob(action))

                        state, reward, done, truncated, _ = env.step(action.item())
                        rewards_episode.append(reward)

                        if done or truncated:
                            break
                else:
                    # -------------------------------------
                    # Opponent plays (random agent)
                    # -------------------------------------
                    action_mask = env.get_action_mask()
                    opponent_action = env.action_space.sample(mask=action_mask)

                    state, reward, done, truncated, _ = env.step(opponent_action)

                    if done or truncated:
                        rewards_episode[-1] = reward
                        break

                    # second opponent move
                    action_mask = env.get_action_mask()
                    opponent_action = env.action_space.sample(mask=action_mask)

                    state, reward, done, truncated, _ = env.step(opponent_action)

            # =================================================
            # OTHER ENVIRONMENTS (TicTacToe, GridWorld, etc.)
            # =================================================
            else:
                state_tensor = torch.tensor(state).float()

                action_probs = reinforce_agent(state_tensor, action_mask)
                dist = Categorical(action_probs)

                action = dist.sample()
                log_probs_episode.append(dist.log_prob(action))

                state, reward, done, truncated, _ = env.step(action.item())
                rewards_episode.append(reward)

        # =========================================================
        # Episode finished → compute learning signal
        # =========================================================
        rewards_history[episode] = np.sum(rewards_episode)

        returns = compute_returns(rewards_episode, gamma)
        loss = compute_policy_loss(log_probs_episode, returns)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history[episode] = loss.item()

        # =========================================================
        # Logging
        # =========================================================
        if episode % 100 == 0:
            recent_rewards = rewards_history[max(0, episode - 100):episode + 1]

            win_rate = np.mean(recent_rewards == 1) * 100
            loss_rate = np.mean(recent_rewards == -1) * 100

            print(
                f"Episode {episode} | "
                f"Win={win_rate:.0f}% Lose={loss_rate:.0f}% | "
                f"Policy Loss={loss_history[episode]:.4f}"
            )

    # =========================================================
    # Plot training dashboard
    # =========================================================
    plot_rl_dashboard(
        rewards_history,
        loss_history,
        algo_name="REINFORCE",
    )

    # =========================================================
    # Save model
    # =========================================================
    model_dir = f"{settings.models_path}/reinforce/{env.unwrapped}"
    os.makedirs(model_dir, exist_ok=True)

    with open(f"{model_dir}/model.pkl", "wb") as f:
        pickle.dump(reinforce_agent, f)

    env.close()

    return reinforce_agent


# =========================================================
# Entry point
# =========================================================
if __name__ == "__main__":
    env = LineWorldEnv()
    reinforce(env, num_episodes=1_000)

    env = GridWorldEnv()
    reinforce(env, num_episodes=10_000)

    env = TicTacToeEnv()
    reinforce(env, num_episodes=10_000)

    env = QuartoEnv()
    reinforce(env, num_episodes=100_000)
