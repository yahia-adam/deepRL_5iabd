import os
import torch
import random
import pickle
import numpy as np
import torch.nn as nn
import gymnasium as gym
from torch import optim
from torch.distributions import Categorical

from deeprl_5iabd.helper import softmax_with_mask, plot_rl_dashboard
from deeprl_5iabd.envs.line_world import LineWorldEnv
from deeprl_5iabd.envs.grid_world import GridWorldEnv
from deeprl_5iabd.envs.tictactoe import TicTacToeEnv
from deeprl_5iabd.envs.quarto import QuartoEnv, Phase
from deeprl_5iabd.config import settings


seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# =========================================================
# Policy / Value networks (identiques à REINFORCE)
# =========================================================
class ActorAgent(nn.Module):
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


class CriticAgent(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_size = np.array(env.observation_space.shape).prod()
        self.network = nn.Sequential(
            nn.Linear(obs_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 1),
        )

    def forward(self, state_tensor):
        return self.network(state_tensor).squeeze(-1)


# =========================================================
# Monte Carlo returns (identique à REINFORCE)
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
# Losses PPO
# =========================================================
def compute_ppo_loss(new_log_probs, old_log_probs, advantages, clip_eps):
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(surr1, surr2).mean()


def compute_critic_loss(values, returns):
    return ((values - returns) ** 2).mean()


# =========================================================
# Training loop
# =========================================================
def ppo(
    env: gym.Env,
    num_episodes: int = 10_000,
    lr: float = 0.001,
    gamma: float = 0.99,
    clip_eps: float = 0.2,
    update_epochs: int = 4,
):

    actor_agent = ActorAgent(env)
    critic_agent = CriticAgent(env)

    actor_optimizer = optim.Adam(actor_agent.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic_agent.parameters(), lr=lr)

    rewards_history = np.zeros(num_episodes)
    loss_history = np.zeros(num_episodes)

    # =========================================================
    # Training loop over episodes
    # =========================================================
    for episode in range(num_episodes):

        # Rollout buffers
        states_episode = []
        actions_episode = []
        masks_episode = []
        old_log_probs_episode = []
        rewards_episode = []

        state, _ = env.reset(seed)
        done = False

        # =====================================================
        # Interaction loop (identique à REINFORCE)
        # =====================================================
        while not done:
            action_mask = env.get_action_mask()

            # =================================================
            # SPECIAL CASE: Quarto environment
            # =================================================
            if isinstance(env, QuartoEnv):

                if env.current_player == env.agent_player:
                    # -------- PLACE --------
                    if env.phase == Phase.PLACE:
                        state_tensor = torch.tensor(state).float()

                        with torch.no_grad():
                            action_probs = actor_agent(state_tensor, action_mask)
                        dist = Categorical(action_probs)
                        action = dist.sample()

                        states_episode.append(state_tensor)
                        actions_episode.append(action)
                        masks_episode.append(action_mask)
                        old_log_probs_episode.append(dist.log_prob(action).detach())

                        state, reward, done, truncated, _ = env.step(action.item())
                        rewards_episode.append(reward)

                        if done or truncated:
                            break

                    # -------- SELECT --------
                    if env.phase == Phase.SELECT:
                        state_tensor = torch.tensor(state).float()

                        with torch.no_grad():
                            action_probs = actor_agent(state_tensor, action_mask)
                        dist = Categorical(action_probs)
                        action = dist.sample()

                        states_episode.append(state_tensor)
                        actions_episode.append(action)
                        masks_episode.append(action_mask)
                        old_log_probs_episode.append(dist.log_prob(action).detach())

                        state, reward, done, truncated, _ = env.step(action.item())
                        rewards_episode.append(reward)

                        if done or truncated:
                            break
                else:
                    # -------------------------------------
                    # Opponent plays (random)
                    # -------------------------------------
                    action_mask = env.get_action_mask()
                    opponent_action = env.action_space.sample(mask=action_mask)
                    state, reward, done, truncated, _ = env.step(opponent_action)

                    if done or truncated:
                        rewards_episode[-1] = reward
                        break

                    action_mask = env.get_action_mask()
                    opponent_action = env.action_space.sample(mask=action_mask)
                    state, reward, done, truncated, _ = env.step(opponent_action)

            # =================================================
            # OTHER ENVIRONMENTS
            # =================================================
            else:
                state_tensor = torch.tensor(state).float()

                with torch.no_grad():
                    action_probs = actor_agent(state_tensor, action_mask)
                dist = Categorical(action_probs)
                action = dist.sample()

                states_episode.append(state_tensor)
                actions_episode.append(action)
                masks_episode.append(action_mask)
                old_log_probs_episode.append(dist.log_prob(action).detach())

                state, reward, done, truncated, _ = env.step(action.item())
                rewards_episode.append(reward)

        # =========================================================
        # Episode fini → PPO update
        # =========================================================
        rewards_history[episode] = np.sum(rewards_episode)

        returns = compute_returns(rewards_episode, gamma)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        old_log_probs_tensor = torch.stack(old_log_probs_episode)
        actions_tensor = torch.stack(actions_episode)

        # K passes d'update sur le rollout collecté
        for _ in range(update_epochs):
            new_log_probs = []
            values = []
            for s, a, m in zip(states_episode, actions_tensor, masks_episode):
                probs = actor_agent(s, m)
                dist = Categorical(probs)
                new_log_probs.append(dist.log_prob(a))
                values.append(critic_agent(s))

            new_log_probs = torch.stack(new_log_probs)
            values = torch.stack(values)

            advantages = returns_tensor - values.detach()
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            actor_loss = compute_ppo_loss(new_log_probs, old_log_probs_tensor, advantages, clip_eps)
            critic_loss = compute_critic_loss(values, returns_tensor)

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

        loss_history[episode] = actor_loss.item()

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
    # Plot + save
    # =========================================================
    exp_name = f"clip={clip_eps} epochs={update_epochs} seed={seed}"
    plot_rl_dashboard(
        rewards_history,
        loss_history,
        algo_name="ppo",
        env_name=env.unwrapped,
        prams=exp_name,
    )

    model_dir = f"{settings.models_path}/ppo/{env.unwrapped}"
    os.makedirs(model_dir, exist_ok=True)
    with open(f"{model_dir}/model.pkl", "wb") as f:
        pickle.dump(actor_agent, f)

    env.close()
    return actor_agent


# =========================================================
# Entry point
# =========================================================
if __name__ == "__main__":
    env = LineWorldEnv()
    ppo(env, num_episodes=10_000)

    env = GridWorldEnv()
    ppo(env, num_episodes=50_000)

    env = TicTacToeEnv()
    ppo(env, num_episodes=100_000)

    env = QuartoEnv()
    ppo(env, num_episodes=100_000)