import os
import pickle
import time
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from deeprl_5iabd.config import settings
from deeprl_5iabd.envs.grid_world import GridWorldEnv
from deeprl_5iabd.envs.line_world import LineWorldEnv
from deeprl_5iabd.envs.tictactoe import TicTacToeEnv
from deeprl_5iabd.envs.quarto import QuartoEnv, Phase
from deeprl_5iabd.agents.dqn import (
    QNetwork,
    apply_line_grid_step_cap,
    capped_eval_max_episode_steps,
    choose_action_epsilon_greedy,
    load_qnet_and_eval,
    set_seed,
)
from deeprl_5iabd.helper import plot_metric


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_prime, next_mask, done):
        self.buffer.append((s, a, r, s_prime, next_mask, done))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        s, a, r, s_prime, next_mask, done = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(s_prime, dtype=np.float32),
            np.array(next_mask, dtype=np.float32),
            np.array(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def ddqn_batch_q_sa_and_td_target(
    q_net: nn.Module,
    target_net: nn.Module,
    X: torch.Tensor,
    X_next: torch.Tensor,
    actions_t: torch.Tensor,
    rewards_t: torch.Tensor,
    dones_t: torch.Tensor,
    masks_t: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_current = q_net(X)
    q_sa = q_current.gather(1, actions_t.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        q_next_online = q_net(X_next)
        neg_inf = torch.full_like(q_next_online, float("-inf"))
        q_next_online_masked = torch.where(masks_t > 0, q_next_online, neg_inf)
        best_next_actions = q_next_online_masked.argmax(dim=1)
        q_next_target = target_net(X_next)
        target_q_next = q_next_target.gather(
            1, best_next_actions.unsqueeze(1)
        ).squeeze(1)
        td_target = rewards_t + gamma * target_q_next * (1.0 - dones_t)

    return q_sa, td_target


def _ddqn_batch_update(
    q_net: QNetwork,
    target_net: QNetwork,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
) -> float:
    states, actions, rewards, next_states, next_masks, dones = replay_buffer.sample(batch_size)

    X = torch.tensor(states)
    X_next = torch.tensor(next_states)
    actions_t = torch.tensor(actions, dtype=torch.long)
    rewards_t = torch.tensor(rewards)
    dones_t = torch.tensor(dones)
    masks_t = torch.tensor(next_masks)
    q_sa, td_target = ddqn_batch_q_sa_and_td_target(
        q_net, target_net, X, X_next, actions_t, rewards_t, dones_t, masks_t, gamma,
    )
    loss = loss_fn(q_sa, td_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def _store_and_train(
    replay_buffer: ReplayBuffer,
    q_net: QNetwork,
    target_net: QNetwork,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    state, action, reward, new_state, next_mask, done,
    batch_size: int,
    gamma: float,
    learning_starts: int,
    train_freq: int,
    global_step: int,
) -> Optional[float]:
    replay_buffer.push(state, action, reward, new_state, next_mask, done)
    if len(replay_buffer) < max(batch_size, learning_starts):
        return None
    if global_step % train_freq != 0:
        return None
    return _ddqn_batch_update(
        q_net, target_net, optimizer, loss_fn,
        replay_buffer, batch_size, gamma,
    )


def ddqn_replay(
    env: gym.Env,
    q_net: QNetwork = None,
    num_episodes: int = 10_000,
    lr: float = 2.5e-4,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.1,
    epsilon_anneal_frac: float = 0.5,
    hidden_size: int = 128,
    target_update_freq: int = 500,
    buffer_capacity: int = 50_000,
    batch_size: int = 32,
    learning_starts: int = 1_000,
    train_freq: int = 4,
    seed: int = 42,
    checkpoints: tuple = (1_000, 10_000, 100_000),
) -> QNetwork:
    set_seed(seed)
    if q_net is None:
        q_net = QNetwork(env, hidden_size=hidden_size)

    agent_name = "ddqn_replay"
    model_dir = f"{settings.models_path}/{agent_name}/{env.unwrapped}/seed_{seed}"
    os.makedirs(model_dir, exist_ok=True)
    plot_dir = f"{settings.training_logs_dir}/{agent_name}/{env.unwrapped}/seed_{seed}"

    target_net = QNetwork(env, hidden_size=hidden_size)
    target_net.load_state_dict(q_net.state_dict())
    for p in target_net.parameters():
        p.requires_grad = False

    optimizer = optim.RMSprop(q_net.parameters(), lr=lr, momentum=0.95)
    loss_fn = nn.MSELoss()
    replay_buffer = ReplayBuffer(buffer_capacity)

    rewards_history = np.zeros(num_episodes)
    loss_history = np.zeros(num_episodes)
    nbr_steps_history = np.zeros(num_episodes, dtype=int)
    time_per_move_history = np.zeros(num_episodes)

    epsilon = epsilon_start
    global_step = 0
    epsilon_anneal_episodes = max(1, int(epsilon_anneal_frac * num_episodes))

    def maybe_update(state, action, reward, new_state, next_mask, done):
        nonlocal global_step
        global_step += 1
        loss = _store_and_train(
            replay_buffer, q_net, target_net, optimizer, loss_fn,
            state, action, reward, new_state, next_mask, done,
            batch_size, gamma, learning_starts, train_freq,
            global_step,
        )
        if global_step % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())
        return loss

    for epoch in range(num_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        ep_rewards: list = []
        ep_losses: list = []
        n_step = 0
        episode_start = time.perf_counter()

        while not terminated and not truncated:
            mask = env.get_action_mask()

            if isinstance(env, QuartoEnv):
                if env.phase == Phase.PLACE:
                    action = choose_action_epsilon_greedy(state, mask, q_net, epsilon)
                    new_state, reward, terminated, truncated, _ = env.step(action)
                    n_step += 1
                    next_mask = env.get_action_mask()
                    loss = maybe_update(
                        state, action, reward, new_state, next_mask,
                        terminated or truncated,
                    )
                    if loss is not None:
                        ep_losses.append(loss)
                    ep_rewards.append(reward)
                    state = new_state
                    if terminated or truncated:
                        break
                    mask = env.get_action_mask()
                if env.phase == Phase.SELECT:
                    action = choose_action_epsilon_greedy(state, mask, q_net, epsilon)
                    new_state, reward, terminated, truncated, _ = env.step(action)
                    n_step += 1
                    if not (terminated or truncated):
                        opp_mask = env.get_action_mask()
                        opp_action = env.action_space.sample(mask=opp_mask)
                        new_state, opp_reward, terminated, truncated, _ = env.step(opp_action)
                        n_step += 1
                        if terminated or truncated:
                            reward = opp_reward
                        else:
                            opp_mask = env.get_action_mask()
                            opp_action = env.action_space.sample(mask=opp_mask)
                            new_state, opp_reward, terminated, truncated, _ = env.step(opp_action)
                            n_step += 1
                            if terminated or truncated:
                                reward = opp_reward
                    next_mask = env.get_action_mask()
                    loss = maybe_update(
                        state, action, reward, new_state, next_mask,
                        terminated or truncated,
                    )
                    if loss is not None:
                        ep_losses.append(loss)
                    ep_rewards.append(reward)
                    state = new_state

            elif isinstance(env, TicTacToeEnv):
                action = choose_action_epsilon_greedy(state, mask, q_net, epsilon)
                new_state, reward, terminated, truncated, _ = env.step(action)
                n_step += 1
                if not (terminated or truncated):
                    opp_mask = env.get_action_mask()
                    opp_action = env.action_space.sample(mask=opp_mask)
                    new_state, reward, terminated, truncated, _ = env.step(opp_action)
                    n_step += 1
                next_mask = env.get_action_mask()
                loss = maybe_update(
                    state, action, reward, new_state, next_mask,
                    terminated or truncated,
                )
                if loss is not None:
                    ep_losses.append(loss)
                ep_rewards.append(reward)
                state = new_state

            else:
                action = choose_action_epsilon_greedy(state, mask, q_net, epsilon)
                new_state, reward, terminated, truncated, _ = env.step(action)
                n_step += 1
                reward, terminated, truncated = apply_line_grid_step_cap(
                    env, n_step, reward, terminated, truncated,
                )
                next_mask = env.get_action_mask()
                loss = maybe_update(
                    state, action, reward, new_state, next_mask,
                    terminated or truncated,
                )
                if loss is not None:
                    ep_losses.append(loss)
                ep_rewards.append(reward)
                state = new_state

        episode_time = time.perf_counter() - episode_start
        time_per_move_history[epoch] = episode_time / max(n_step, 1)
        nbr_steps_history[epoch] = n_step
        rewards_history[epoch] = float(np.sum(ep_rewards))
        loss_history[epoch] = float(np.mean(ep_losses) if ep_losses else 0.0)
        # plus le nombre d'epochs est élevé, plus frac est proche de 1.0
        frac = min(1.0, (epoch + 1) / epsilon_anneal_episodes)
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            recent = rewards_history[max(0, epoch - 99):epoch + 1]
            recent_tpm = time_per_move_history[max(0, epoch - 99):epoch + 1]
            win_rate = float(np.mean(recent == 1) * 100)
            loss_rate = float(np.mean(recent == -1) * 100)
            print(
                f"[{agent_name} | {env.unwrapped} | seed={seed}] Episode {epoch + 1} | "
                f"Win={win_rate:.0f}% Lose={loss_rate:.0f}% | buf={len(replay_buffer)} | "
                f"Loss={loss_history[epoch]:.4f} | "
                f"Time/move={np.mean(recent_tpm) * 1000:.2f}ms | ε={epsilon:.3f}"
            )

        if (epoch + 1) in checkpoints:
            with open(f"{model_dir}/policy_{agent_name}_{epoch + 1}.pkl", "wb") as f:
                pickle.dump(q_net.state_dict(), f)
            print(f"Model saved: {model_dir}/policy_{agent_name}_{epoch + 1}.pkl")

    os.makedirs(plot_dir, exist_ok=True)
    plot_metric(
        values=rewards_history, save_dir=plot_dir, window_size=100,
        exp_name=f"{agent_name}_env_{env.unwrapped}", metric_name="winrate",
    )
    plot_metric(
        values=loss_history, save_dir=plot_dir, window_size=100,
        exp_name=f"training_loss_{agent_name}_env_{env.unwrapped}", metric_name="loss",
    )
    plot_metric(
        values=nbr_steps_history, save_dir=plot_dir, window_size=100,
        exp_name=f"nbr_steps_{agent_name}_env_{env.unwrapped}", metric_name="nbr_steps",
    )
    plot_metric(
        values=time_per_move_history, save_dir=plot_dir, window_size=100,
        exp_name=f"time_per_move_{agent_name}_env_{env.unwrapped}", metric_name="time_per_move",
    )
    with open(f"{model_dir}/policy_{agent_name}_{num_episodes}.pkl", "wb") as f:
        pickle.dump(q_net.state_dict(), f)
    env.close()
    return q_net


def eval_ddqn_replay(
    env: gym.Env,
    num_episodes: int = 1_000,
    model_name: str = "policy_ddqn_replay_10000.pkl",
    seed: int = 42,
    hidden_size: int = 128,
    max_episode_steps: int = 10_000,
) -> dict:
    max_episode_steps = capped_eval_max_episode_steps(env, max_episode_steps)

    return load_qnet_and_eval(
        env=env,
        q_network_class=QNetwork,
        model_name=model_name,
        seed=seed,
        hidden_size=hidden_size,
        algo_subdir="ddqn_replay",
        num_episodes=num_episodes,
        max_episode_steps=max_episode_steps,
        log_tag="EVAL DDQN+Replay",
    )


_ENV_TRAIN_CONFIGS = {
    "lineworld": (
        LineWorldEnv,
        dict(
            lr=2.5e-4,
            hidden_size=16,
            epsilon_anneal_frac=0.1,
            epsilon_end=0.02,
            gamma=0.9,
            buffer_capacity=5_000,
            batch_size=32,
            learning_starts=200,
            target_update_freq=200,
            train_freq=4,
        ),
    ),
    "gridworld": (
        GridWorldEnv,
        dict(
            lr=2.5e-4,
            hidden_size=32,
            epsilon_anneal_frac=0.3,
            epsilon_end=0.1,
            buffer_capacity=20_000,
            batch_size=32,
            learning_starts=500,
            target_update_freq=500,
            train_freq=4,
        ),
    ),
    "tictactoe": (
        TicTacToeEnv,
        dict(
            lr=2.5e-4,
            hidden_size=128,
            epsilon_anneal_frac=0.5,
            epsilon_end=0.1,
            buffer_capacity=50_000,
            batch_size=32,
            learning_starts=1_000,
            target_update_freq=500,
            train_freq=4,
        ),
    ),
    "quarto": (
        QuartoEnv,
        dict(
            lr=2.5e-4,
            hidden_size=256,
            epsilon_anneal_frac=0.6,
            epsilon_end=0.1,
            buffer_capacity=100_000,
            batch_size=32,
            learning_starts=2_000,
            target_update_freq=1_000,
            train_freq=4,
        ),
    ),
}


def _train_single_env_from_cli(env_name: str, episodes: int, seed: int) -> None:
    EnvCls, train_kw = _ENV_TRAIN_CONFIGS[env_name]
    print(f"TRAIN DDQN+Replay {EnvCls.__name__} | episodes={episodes} | seed={seed}")
    env = EnvCls()
    try:
        ddqn_replay(
            env,
            num_episodes=episodes,
            seed=seed,
            checkpoints=(1_000, 10_000, 100_000),
            **train_kw,
        )
    finally:
        env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DDQN+Replay on a selected environment.")
    parser.add_argument("--env", choices=tuple(_ENV_TRAIN_CONFIGS.keys()), required=True)
    parser.add_argument("--episodes", "-n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _train_single_env_from_cli(args.env, args.episodes, args.seed)