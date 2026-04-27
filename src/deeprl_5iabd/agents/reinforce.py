import os
import json
import time
import torch
import random
import pickle
import numpy as np
import torch.nn as nn
import gymnasium as gym
from torch import optim
from torch.distributions import Categorical
from gymnasium.wrappers import RecordVideo

from deeprl_5iabd.helper import softmax_with_mask, plot_metric, plot_trace
from deeprl_5iabd.envs.line_world import LineWorldEnv
from deeprl_5iabd.envs.grid_world import GridWorldEnv
from deeprl_5iabd.envs.tictactoe import TicTacToeEnv
from deeprl_5iabd.envs.quarto import QuartoEnv
from deeprl_5iabd.config import settings


SEEDS = (42, 123, 7)


def set_seed(seed: int):
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
    num_episodes: int = 100_000,
    lr: float = 0.001,
    gamma: float = 0.99,
    use_mean_baseline: bool = True,
    use_critic_baseline: bool = True,
    checkpoints: tuple = (1_000, 10_000, 100_000),
    seed: int = 42,
):
    set_seed(seed)

    reinforce_agent = ActorAgent(env)
    optimizer = optim.Adam(reinforce_agent.parameters(), lr=lr)

    if use_critic_baseline:
        critic_agent = CriticAgent(env)
        critic_optimizer = optim.Adam(critic_agent.parameters(), lr=lr)

    rewards_history = np.zeros(num_episodes + 1)
    loss_history = np.zeros(num_episodes + 1)
    nbr_steps_history = np.zeros(num_episodes + 1)
    time_per_move_history = np.zeros(num_episodes + 1)

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

        episode_start = time.perf_counter()

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

        episode_time = time.perf_counter() - episode_start
        time_per_move_history[episode] = episode_time / max(n_step, 1)

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
            recent_tpm = time_per_move_history[max(1, episode - 100):episode + 1]
            win_rate = np.mean(recent_rewards == 1) * 100
            loss_rate = np.mean(recent_rewards == -1) * 100
            print(
                f"[{agent_name} | {env.unwrapped} | seed={seed}] Episode {episode} | "
                f"Win={win_rate:.0f}% Lose={loss_rate:.0f}% | "
                f"Policy Loss={loss_history[episode]:.4f} | "
                f"Time/move={np.mean(recent_tpm) * 1000:.2f}ms"
            )

        if episode in checkpoints:
            model_dir = f"{settings.models_path}/reinforce/{env.unwrapped}/seed_{seed}"
            os.makedirs(model_dir, exist_ok=True)
            with open(f"{model_dir}/policy_{agent_name}_{episode}.pkl", "wb") as f:
                pickle.dump(reinforce_agent.state_dict(), f)
            print(f"Model saved: {model_dir}/policy_{agent_name}_{episode}.pkl")

    plot_dir = f"{settings.training_logs_dir}/reinforce/{env.unwrapped}/seed_{seed}"

    plot_metric(values=rewards_history, save_dir=plot_dir, window_size=100,
                exp_name=f"{agent_name}_env_{env.unwrapped}", metric_name="winrate")
    plot_metric(values=loss_history, save_dir=plot_dir, window_size=100,
                exp_name=f"training_loss_{agent_name}_env_{env.unwrapped}", metric_name="loss")
    plot_metric(values=nbr_steps_history, save_dir=plot_dir, window_size=100,
                exp_name=f"nbr_steps_{agent_name}_env_{env.unwrapped}", metric_name="nbr_steps")
    plot_metric(values=time_per_move_history, save_dir=plot_dir, window_size=100,
                exp_name=f"time_per_move_{agent_name}_env_{env.unwrapped}", metric_name="time_per_move")

    return reinforce_agent


def eval_agent(env, num_episodes=1_000, model_name="policy_reinforce_no_baseline_1000.pkl", seed: int = 42):
    rewards_history = np.zeros(num_episodes)
    n_steps_history = np.zeros(num_episodes, dtype=int)

    agent = ActorAgent(env)
    with open(f"{settings.models_path}/reinforce/{env.unwrapped}/seed_{seed}/{model_name}", "rb") as f:
        state_dict = pickle.load(f)
    agent.load_state_dict(state_dict)
    agent.eval()

    is_multi = getattr(env, "is_multi_player", False)

    for i in range(num_episodes):
        n_step = 0
        with torch.no_grad():
            state, _ = env.reset()
            done = False
            truncated = False
            reward = 0.0

            while not (done or truncated):

                if is_multi:
                    while not (done or truncated) and env.current_player != env.agent_player:
                        state, reward, done, truncated, _ = opponent_step(env)
                        n_step += 1

                while not (done or truncated) and (not is_multi or env.current_player == env.agent_player):
                    action_mask = env.get_action_mask()
                    state_tensor = torch.tensor(state).float()

                    action_probs = agent(state_tensor, action_mask)
                    dist = Categorical(action_probs)
                    action = dist.sample().item()

                    state, reward, done, truncated, _ = env.step(action)
                    n_step += 1

        rewards_history[i] = reward
        n_steps_history[i] = n_step

    plot_trace(rewards_history, model_name)

    # parse: policy_<agent_name>_<checkpoint>.pkl
    base = model_name.replace("policy_", "").replace(".pkl", "")
    agent_name, checkpoint_str = base.rsplit("_", 1)
    checkpoint = int(checkpoint_str)

    results = {
        "env": str(env.unwrapped),
        "agent": agent_name,
        "checkpoint": checkpoint,
        "seed": seed,
        "num_episodes": num_episodes,
        "summary": {
            "mean_reward": float(np.mean(rewards_history)),
            "win_rate": float(np.mean(rewards_history == 1)),
            "loss_rate": float(np.mean(rewards_history == -1)),
            "draw_rate": float(np.mean(rewards_history == 0)),
            "mean_steps": float(np.mean(n_steps_history)),
            "std_steps": float(np.std(n_steps_history)),
        },
        "episodes": [
            {"episode": int(i), "reward": float(r), "n_steps": int(s)}
            for i, (r, s) in enumerate(zip(rewards_history, n_steps_history))
        ],
    }

    json_dir = f"{settings.training_logs_dir}/reinforce/{env.unwrapped}/seed_{seed}/eval"
    os.makedirs(json_dir, exist_ok=True)
    json_path = f"{json_dir}/{agent_name}_{checkpoint}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    s = results["summary"]
    print(
        f"[EVAL {env.unwrapped}/{agent_name}@{checkpoint} | seed={seed}] "
        f"win={s['win_rate']:.2%} loss={s['loss_rate']:.2%} draw={s['draw_rate']:.2%} "
        f"mean_steps={s['mean_steps']:.2f} -> {json_path}"
    )
    return results


def train_all_baseline_for_env(env, seed):
    reinforce(env, num_episodes=100_000, use_mean_baseline=False, use_critic_baseline=False, seed=seed)
    reinforce(env, num_episodes=100_000, use_mean_baseline=True,  use_critic_baseline=False, seed=seed)
    reinforce(env, num_episodes=100_000, use_mean_baseline=False, use_critic_baseline=True,  seed=seed)


def eval_all_models_for_env(env, seed):
    for n in (1_000, 10_000, 100_000):
        for agent_name in (
            "reinforce_no_baseline",
            "reinforce_mean_baseline",
            "reinforce_critic_baseline",
        ):
            eval_agent(env, num_episodes=1_000, model_name=f"policy_{agent_name}_{n}.pkl", seed=seed)


def wrap_video(env, mode, seed, epideo_num_trigger):
    video_env = RecordVideo(
        env,
        video_folder=f"{settings.videos_dir}/reinforce/{env.unwrapped}/seed_{seed}/{mode}/",
        episode_trigger=lambda ep: ep % epideo_num_trigger == 0,
    )
    video_env.state_id = env.state_id
    video_env.get_action_mask = env.get_action_mask
    video_env.agent_player = env.agent_player
    type(video_env).current_player = property(
        lambda self: env.current_player,
        lambda self, v: setattr(env, 'current_player', v)
    )
    return video_env


if __name__ == "__main__":
    env_classes = [LineWorldEnv, GridWorldEnv, TicTacToeEnv, QuartoEnv]

    for seed in SEEDS:
        for EnvCls in env_classes:
            # ---- TRAIN ----
            env_train = EnvCls(render_mode="rgb_array")
            video_env_train = wrap_video(env_train, "train", seed, 10_000)
            print(f"\n{'=' * 60}\nTRAIN {env_train.unwrapped} | seed={seed}\n{'=' * 60}")
            train_all_baseline_for_env(video_env_train, seed)
            video_env_train.close()

            # ---- EVAL ----
            env_eval = EnvCls(render_mode="rgb_array")
            video_env_eval = wrap_video(env_eval, "eval", seed, 100)
            print(f"\n{'=' * 60}\nEVAL {env_eval.unwrapped} | seed={seed}\n{'=' * 60}")
            eval_all_models_for_env(video_env_eval, seed)
            video_env_eval.close()