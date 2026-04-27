import os
import torch
import random
import pickle
import numpy as np
import torch.nn as nn
import gymnasium as gym
from torch import optim
from torch.distributions import Categorical
from gymnasium.wrappers import RecordVideo

from deeprl_5iabd.helper import softmax_with_mask, plot_metric
from deeprl_5iabd.envs.line_world import LineWorldEnv
from deeprl_5iabd.envs.grid_world import GridWorldEnv
from deeprl_5iabd.envs.tictactoe import TicTacToeEnv
from deeprl_5iabd.envs.quarto import QuartoEnv
from deeprl_5iabd.config import settings


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

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

def compute_ppo_loss(new_log_probs, old_log_probs, advantages, clip_eps):
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(surr1, surr2).mean()


def compute_critic_loss(values, returns):
    return ((values - returns) ** 2).mean()


def opponent_step(env):
    mask = env.get_action_mask()
    action = env.action_space.sample(mask=mask)
    return env.step(action)


def ppo(
    env: gym.Env,
    num_episodes: int = 10_000,
    lr: float = 0.001,
    gamma: float = 0.99,
    clip_eps: float = 0.2,
    update_epochs: int = 4,
    rollout_size: int = 16,
):
    actor_agent = ActorAgent(env)
    critic_agent = CriticAgent(env)

    actor_optimizer = optim.Adam(actor_agent.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic_agent.parameters(), lr=lr)

    rewards_history = np.zeros(num_episodes + 1)
    loss_history = np.zeros(num_episodes + 1)
    nbr_steps_history = np.zeros(num_episodes + 1)

    agent_name = f"ppo_clip={clip_eps}_epochs={update_epochs}_rollout={rollout_size}"

    is_multi = getattr(env, "is_multi_player", False)

    buf_states, buf_actions, buf_masks = [], [], []
    buf_old_log_probs, buf_returns = [], []
    last_actor_loss = 0.0

    for episode in range(1, num_episodes + 1):

        states_episode = []
        actions_episode = []
        masks_episode = []
        old_log_probs_episode = []

        n_step = 0
        final_reward = 0.0

        state, _ = env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            if is_multi:
                while not (done or truncated) and env.current_player != env.agent_player:
                    state, final_reward, done, truncated, _ = opponent_step(env)
                    n_step += 1

            while not (done or truncated) and (not is_multi or env.current_player == env.agent_player):
                action_mask = env.get_action_mask()
                state_tensor = torch.tensor(state).float()

                with torch.no_grad():
                    action_probs = actor_agent(state_tensor, action_mask)
                dist = Categorical(action_probs)
                action = dist.sample()

                states_episode.append(state_tensor)
                actions_episode.append(action)
                masks_episode.append(action_mask)
                old_log_probs_episode.append(dist.log_prob(action).detach())

                state, final_reward, done, truncated, _ = env.step(action.item())
                n_step += 1

        rewards_history[episode] = final_reward
        nbr_steps_history[episode] = n_step

        n = len(states_episode)
        if n > 0:
            rewards_episode = [0.0] * n
            rewards_episode[-1] = final_reward

            ep_returns = compute_returns(rewards_episode, gamma)
            buf_states.extend(states_episode)
            buf_actions.extend(actions_episode)
            buf_masks.extend(masks_episode)
            buf_old_log_probs.extend(old_log_probs_episode)
            buf_returns.extend(ep_returns)

        if episode % rollout_size == 0 and len(buf_states) > 0:

            returns_tensor = torch.tensor(buf_returns, dtype=torch.float32)
            old_log_probs_tensor = torch.stack(buf_old_log_probs)
            actions_tensor = torch.stack(buf_actions)

            for _ in range(update_epochs):
                new_log_probs = []
                values = []
                for s, a, m in zip(buf_states, actions_tensor, buf_masks):
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

                last_actor_loss = actor_loss.item()

            buf_states.clear()
            buf_actions.clear()
            buf_masks.clear()
            buf_old_log_probs.clear()
            buf_returns.clear()

        loss_history[episode] = last_actor_loss

        if episode % 100 == 0:
            recent_rewards = rewards_history[max(1, episode - 100):episode + 1]
            win_rate = np.mean(recent_rewards == 1) * 100
            loss_rate = np.mean(recent_rewards == -1) * 100
            print(
                f"Episode {episode} | "
                f"Win={win_rate:.0f}% Lose={loss_rate:.0f}% | "
                f"Policy Loss={loss_history[episode]:.4f}"
            )

        if episode in (1_000, 10_000, 100_000):
            model_dir = f"{settings.models_path}/ppo/{env.unwrapped}"
            os.makedirs(model_dir, exist_ok=True)
            with open(f"{model_dir}/policy_{agent_name}_{episode}.pkl", "wb") as f:
                pickle.dump(actor_agent.state_dict(), f)
            print(f"Model saved: {model_dir}/policy_{agent_name}_{episode}.pkl")

    winrate_path = plot_metric(
        values=rewards_history,
        save_dir=f"{settings.training_logs_dir}/ppo/{env.unwrapped}",
        window_size=100,
        exp_name=f"{agent_name}_env_{env.unwrapped}",
        metric_name="winrate"
    )
    print("Winrate path: ", winrate_path)

    loss_path = plot_metric(
        values=loss_history,
        save_dir=f"{settings.training_logs_dir}/ppo/{env.unwrapped}",
        window_size=100,
        exp_name=f"training_loss_{agent_name}_env_{env.unwrapped}",
        metric_name="loss"
    )
    print("Loss path: ", loss_path)

    nbr_steps_path = plot_metric(
        values=nbr_steps_history,
        save_dir=f"{settings.training_logs_dir}/ppo/{env.unwrapped}",
        window_size=100,
        exp_name=f"nbr_steps_{agent_name}_env_{env.unwrapped}",
        metric_name="nbr_steps"
    )
    print("Nbr steps path: ", nbr_steps_path)

    return actor_agent

def eval_agent(env, num_episodes=100, model_name="policy_ppo_clip=0.2_epochs=4_rollout=16_1000.pkl"):
    rewards_history = np.zeros(num_episodes)

    agent = ActorAgent(env)
    with open(f"{settings.models_path}/ppo/{env.unwrapped}/{model_name}", "rb") as f:
        state_dict = pickle.load(f)
    agent.load_state_dict(state_dict)
    agent.eval()

    is_multi = getattr(env, "is_multi_player", False)

    for i in range(num_episodes):
        with torch.no_grad():
            state, _ = env.reset()
            done = False
            truncated = False
            reward = 0.0

            while not (done or truncated):

                if is_multi:
                    while not (done or truncated) and env.current_player != env.agent_player:
                        state, reward, done, truncated, _ = opponent_step(env)

                while not (done or truncated) and (not is_multi or env.current_player == env.agent_player):
                    action_mask = env.get_action_mask()
                    state_tensor = torch.tensor(state).float()

                    action_probs = agent(state_tensor, action_mask)
                    dist = Categorical(action_probs)
                    action = dist.sample().item()

                    state, reward, done, truncated, _ = env.step(action)

        rewards_history[i] = reward

    plot_metric(
        values=rewards_history,
        save_dir=f"{settings.evaluation_logs_dir}/ppo/{env.unwrapped}",
        window_size=100,
        exp_name=f"ppo_eval_{env.unwrapped}",
        metric_name="winrate"
    )


def train_for_env(env, clip_eps=0.2, update_epochs=4, rollout_size=16):
    ppo(env, num_episodes=10_000,
        clip_eps=clip_eps, update_epochs=update_epochs, rollout_size=rollout_size)


def eval_all_models_for_env(env, clip_eps=0.2, update_epochs=4, rollout_size=16):
    tag = f"ppo_clip={clip_eps}_epochs={update_epochs}_rollout={rollout_size}"
    for n in [1000, 10_000, 100_000]:
        eval_agent(env, num_episodes=10_000, model_name=f"policy_{tag}_{n}.pkl")


if __name__ == "__main__":
    MODE = "train"  # "train" or "eval"

    env = LineWorldEnv(render_mode="rgb_array")
    # env = GridWorldEnv(render_mode="rgb_array")
    # env = TicTacToeEnv(render_mode="rgb_array")
    # env = QuartoEnv(render_mode="rgb_array")

    video_env = RecordVideo(
        env,
        video_folder=f"{settings.videos_dir}/ppo/{env.unwrapped}/{MODE}/",
        episode_trigger=lambda ep: ep % 1_000 == 0,
    )
    video_env.state_id = env.state_id
    video_env.get_action_mask = env.get_action_mask
    video_env.agent_player = env.agent_player
    type(video_env).current_player = property(
        lambda self: env.current_player,
        lambda self, v: setattr(env, 'current_player', v)
    )
    if MODE == "train":
        train_for_env(video_env)
    else:
        eval_all_models_for_env(video_env)
    video_env.close()