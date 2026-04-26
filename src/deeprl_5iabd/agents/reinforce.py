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
        observation_size = np.array(env.observation_space.shape).prod()
        self.network = nn.Sequential(
            nn.Linear(observation_size, 120),
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


def compute_policy_loss(log_probs, returns, baseline=0, critic_values=None):
    loss = 0
    if critic_values is not None:
        for log_prob_t, return_t, critic_value in zip(log_probs, returns, critic_values):
            advantage = return_t - critic_value.detach()
            loss += -log_prob_t * advantage
    else:
        for log_prob_t, return_t in zip(log_probs, returns):
            advantage = return_t - baseline
            loss += -log_prob_t * advantage
    return loss


def compute_critic_loss(critic_values, returns):
    total_loss = 0
    for c_value, re in zip(critic_values, returns):
        total_loss += (c_value - re) ** 2
    total_loss = total_loss / len(returns)
    return total_loss

def opponent_step(env):
    mask = env.get_action_mask()
    action = env.action_space.sample(mask=mask)
    return env.step(action)


def reinforce(
    env: gym.Env,
    num_episodes: int = 10_000,
    lr: float = 0.001,
    gamma: float = 0.99,
    use_mean_baseline: bool = True,
    use_critic_baseline: bool = True,
):
    reinforce_agent = ActorAgent(env)
    optimizer = optim.Adam(reinforce_agent.parameters(), lr=lr)

    if use_critic_baseline:
        critic_agent = CriticAgent(env)
        critic_optimizer = optim.Adam(critic_agent.parameters(), lr=lr)

    rewards_history = np.zeros(num_episodes + 1)
    loss_history = np.zeros(num_episodes + 1)
    nbr_steps_history = np.zeros(num_episodes + 1)

    is_multi = getattr(env, "is_multi_player", False)

    agent_name = (
        "reinforce_critic_baseline" if use_critic_baseline
        else "reinforce_mean_baseline" if use_mean_baseline
        else "reinforce_no_baseline"
    )

    for episode in range(1, num_episodes + 1):

        log_probs_episode = []
        episode_critic_values = []
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

                action_probs = reinforce_agent(state_tensor, action_mask)
                dist = Categorical(action_probs)
                action = dist.sample()

                log_probs_episode.append(dist.log_prob(action))

                if use_critic_baseline:
                    value = critic_agent(state_tensor)
                    episode_critic_values.append(value)

                state, final_reward, done, truncated, _ = env.step(action.item())
                n_step += 1
                
        rewards_history[episode] = final_reward
        nbr_steps_history[episode] = n_step


        rewards_episode = [0.0] * n_step
        rewards_episode[-1] = final_reward
        returns = compute_returns(rewards_episode, gamma)

        if use_critic_baseline:
            loss = compute_policy_loss(log_probs_episode, returns,
                                       critic_values=episode_critic_values)
        else:
            baseline = np.mean(returns) if use_mean_baseline else 0
            loss = compute_policy_loss(log_probs_episode, returns, baseline=baseline)

        if use_critic_baseline:
            critic_loss = compute_critic_loss(episode_critic_values, returns)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history[episode] = loss.item()

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
            model_dir = f"{settings.models_path}/reinforce/{env.unwrapped}"
            os.makedirs(model_dir, exist_ok=True)
            with open(f"{model_dir}/policy_{agent_name}_{episode}.pkl", "wb") as f:
                pickle.dump(reinforce_agent.state_dict(), f)
            print(f"Model saved: {model_dir}/policy_{agent_name}_{episode}.pkl")

    winrate_path = plot_metric(
        values=rewards_history,
        save_dir=f"{settings.training_logs_dir}/reinforce/{env.unwrapped}",
        window_size=100,
        exp_name=f"{agent_name}_env_{env.unwrapped}",
        metric_name="winrate"
    )
    print("Winrate path: ", winrate_path)

    loss_path = plot_metric(
        values=loss_history,
        save_dir=f"{settings.training_logs_dir}/reinforce/{env.unwrapped}",
        window_size=100,
        exp_name=f"training_loss_{agent_name}_env_{env.unwrapped}",
        metric_name="loss"
    )
    print("Loss path: ", loss_path)

    nbr_steps_path = plot_metric(
        values=nbr_steps_history,
        save_dir=f"{settings.training_logs_dir}/reinforce/{env.unwrapped}",
        window_size=100,
        exp_name=f"nbr_steps_{agent_name}_env_{env.unwrapped}",
        metric_name="nbr_steps"
    )
    print("Nbr steps path: ", nbr_steps_path)

    return reinforce_agent


def eval_agent(env, num_episodes=100, model_name="policy_reinforce_no_baseline_1000.pkl"):
    rewards_history = np.zeros(num_episodes)

    return reinforce_agent



def eval_agent(env, num_episodes=100, model_name="policy_reinforce_no_baseline_1000.pkl"):
    rewards_history = np.zeros(num_episodes)

    agent = ActorAgent(env)
    with open(f"{settings.models_path}/reinforce/{env.unwrapped}/{model_name}", "rb") as f:
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
        save_dir=f"{settings.evaluation_logs_dir}/reinforce/{env.unwrapped}",
        window_size=100,
        exp_name=f"reinforce_eval_{env.unwrapped}",
        metric_name="winrate"
    )


def train_all_baseline_for_env(env):
    reinforce(env, num_episodes=10_000, use_mean_baseline=False, use_critic_baseline=False)
    reinforce(env, num_episodes=10_000, use_mean_baseline=True,  use_critic_baseline=False)
    reinforce(env, num_episodes=10_000, use_mean_baseline=False, use_critic_baseline=True)


def eval_all_models_for_env(env):
    for n in [1000, 10_000, 100_000]:
        eval_agent(env, num_episodes=10_000, model_name=f"policy_reinforce_no_baseline_{n}.pkl")
        eval_agent(env, num_episodes=10_000, model_name=f"policy_reinforce_mean_baseline_{n}.pkl")
        eval_agent(env, num_episodes=10_000, model_name=f"policy_reinforce_critic_baseline_{n}.pkl")


if __name__ == "__main__":
    MODE = "eval" # "train" or "eval"

    env = LineWorldEnv(render_mode="rgb_array")
    # env = GridWorldEnv(render_mode="rgb_array")
    # env = TicTacToeEnv(render_mode="rgb_array")
    # env = QuartoEnv(render_mode="rgb_array")

    video_env = RecordVideo(
        env,
        video_folder=f"{settings.videos_dir}/reinforce/{env.unwrapped}/{MODE}/",
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
        train_all_baseline_for_env(video_env)
    else:
        eval_all_models_for_env(video_env)
    video_env.close()
