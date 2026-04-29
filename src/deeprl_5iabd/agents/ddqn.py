import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from deeprl_5iabd.config import settings
from deeprl_5iabd.agents.dqn import (
    QNetwork,
    apply_line_grid_step_cap,
    capped_eval_max_episode_steps,
    choose_action_epsilon_greedy,
    load_qnet_and_eval,
    set_seed,
)
from deeprl_5iabd.envs.grid_world import GridWorldEnv
from deeprl_5iabd.envs.line_world import LineWorldEnv
from deeprl_5iabd.envs.quarto import QuartoEnv, Phase
from deeprl_5iabd.envs.tictactoe import TicTacToeEnv
from deeprl_5iabd.helper import plot_metric


# TD update : next action depuis l'online network, valeur depuis le target network (Double DQN).
def _ddqn_update(
    q_net: QNetwork,
    target_net: QNetwork,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    state: np.ndarray,
    action: int,
    reward: float,
    next_state: np.ndarray,
    next_mask: np.ndarray,
    done: bool,
    gamma: float,
) -> float:
    x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    x_next = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
    q_sa = q_net(x)[0, action]

    with torch.no_grad():
        if done:
            target_q_next = torch.tensor(0.0)
        else:
            q_next_online = q_net(x_next)[0].numpy()
            available_next = np.where(np.asarray(next_mask) == 1)[0]
            if len(available_next) == 0:
                target_q_next = torch.tensor(0.0)
            else:
                masked_online = np.full_like(q_next_online, -np.inf)
                masked_online[available_next] = q_next_online[available_next]
                best_next_action = int(np.argmax(masked_online))
                q_next_target = target_net(x_next)[0]
                target_q_next = q_next_target[best_next_action]
        td_target = reward + gamma * target_q_next

    loss = loss_fn(q_sa, td_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# Double DQN avec synchro du target network ; epsilon linéairement amorti sur l'entraînement.
def ddqn(
    env: gym.Env,
    q_net: QNetwork = None,
    num_episodes: int = 10_000,
    lr: float = 2.5e-4,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.1,
    epsilon_anneal_frac: float = 0.5,
    hidden_size: int = 128,
    target_update_freq: int = 100,
    seed: int = 42,
    checkpoints: tuple = (1_000, 10_000, 100_000),
) -> QNetwork:
    set_seed(seed)
    if q_net is None:
        q_net = QNetwork(env, hidden_size=hidden_size)

    agent_name = "ddqn"
    model_dir = f"{settings.models_path}/{agent_name}/{env.unwrapped}/seed_{seed}"
    os.makedirs(model_dir, exist_ok=True)
    plot_dir = f"{settings.training_logs_dir}/{agent_name}/{env.unwrapped}/seed_{seed}"

    target_net = QNetwork(env, hidden_size=hidden_size)
    target_net.load_state_dict(q_net.state_dict())
    for p in target_net.parameters():
        p.requires_grad = False

    global_step = 0
    optimizer = optim.RMSprop(q_net.parameters(), lr=lr, momentum=0.95)
    loss_fn = nn.MSELoss()

    epsilon_anneal_episodes = max(1, int(epsilon_anneal_frac * num_episodes))

    rewards_history = np.zeros(num_episodes)
    loss_history = np.zeros(num_episodes)
    nbr_steps_history = np.zeros(num_episodes, dtype=int)
    time_per_move_history = np.zeros(num_episodes)
    epsilon = epsilon_start

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
                    loss = _ddqn_update(
                        q_net, target_net, optimizer, loss_fn,
                        state, action, reward, new_state, next_mask,
                        terminated or truncated, gamma,
                    )
                    global_step += 1
                    if global_step % target_update_freq == 0:
                        target_net.load_state_dict(q_net.state_dict())
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
                    loss = _ddqn_update(
                        q_net, target_net, optimizer, loss_fn,
                        state, action, reward, new_state, next_mask,
                        terminated or truncated, gamma,
                    )
                    global_step += 1
                    if global_step % target_update_freq == 0:
                        target_net.load_state_dict(q_net.state_dict())
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
                loss = _ddqn_update(
                    q_net, target_net, optimizer, loss_fn,
                    state, action, reward, new_state, next_mask,
                    terminated or truncated, gamma,
                )
                global_step += 1
                if global_step % target_update_freq == 0:
                    target_net.load_state_dict(q_net.state_dict())
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
                loss = _ddqn_update(
                    q_net, target_net, optimizer, loss_fn,
                    state, action, reward, new_state, next_mask,
                    terminated or truncated, gamma,
                )
                global_step += 1
                if global_step % target_update_freq == 0:
                    target_net.load_state_dict(q_net.state_dict())
                ep_losses.append(loss)
                ep_rewards.append(reward)
                state = new_state

        episode_time = time.perf_counter() - episode_start
        time_per_move_history[epoch] = episode_time / max(n_step, 1)
        nbr_steps_history[epoch] = n_step
        rewards_history[epoch] = float(np.sum(ep_rewards))
        loss_history[epoch] = float(np.mean(ep_losses) if ep_losses else 0.0)

        frac = min(1.0, (epoch + 1) / epsilon_anneal_episodes)
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            recent = rewards_history[max(0, epoch - 99):epoch + 1]
            recent_tpm = time_per_move_history[max(0, epoch - 99):epoch + 1]
            win_rate = float(np.mean(recent == 1) * 100)
            loss_rate = float(np.mean(recent == -1) * 100)
            print(
                f"[{agent_name} | {env.unwrapped} | seed={seed}] Episode {epoch + 1} | "
                f"Win={win_rate:.0f}% Lose={loss_rate:.0f}% | "
                f"Loss={loss_history[epoch]:.4f} | "
                f"Time/move={np.mean(recent_tpm) * 1000:.2f}ms | epsilon={epsilon:.3f}"
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


# Charger le checkpoint DDQN et écrire l'eval JSON via load_qnet_and_eval (shared).
def eval_ddqn(
    env: gym.Env,
    num_episodes: int = 1_000,
    model_name: str = "policy_ddqn_10000.pkl",
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
        algo_subdir="ddqn",
        num_episodes=num_episodes,
        max_episode_steps=max_episode_steps,
        log_tag="EVAL DDQN",
    )


_NUM_EPISODES_100K = 100_000
_CHECKPOINTS_100K = (1_000, 10_000, 100_000)
_MODEL_PREFIX_100K = "policy_ddqn"

# Pilote long : planification linéaire d'epsilon par env (fraction des 100k épisodes).
_TRAIN_EVAL_100K_CONFIGS = [
    (
        LineWorldEnv,
        dict(
            lr=2.5e-4,
            hidden_size=16,
            epsilon_anneal_frac=0.05,
            epsilon_end=0.02,
            gamma=0.9,
        ),
    ),
    (
        GridWorldEnv,
        dict(
            lr=2.5e-4,
            hidden_size=32,
            epsilon_anneal_frac=0.3,
            epsilon_end=0.1,
        ),
    ),
    (
        TicTacToeEnv,
        dict(
            lr=2.5e-4,
            hidden_size=128,
            epsilon_anneal_frac=0.6,
            epsilon_end=0.1,
        ),
    ),
    (
        QuartoEnv,
        dict(
            lr=2.5e-4,
            hidden_size=256,
            epsilon_anneal_frac=0.7,
            epsilon_end=0.1,
        ),
    ),
]

_ENV_TRAIN_CONFIGS = {
    "lineworld": (LineWorldEnv, _TRAIN_EVAL_100K_CONFIGS[0][1]),
    "gridworld": (GridWorldEnv, _TRAIN_EVAL_100K_CONFIGS[1][1]),
    "tictactoe": (TicTacToeEnv, _TRAIN_EVAL_100K_CONFIGS[2][1]),
    "quarto": (QuartoEnv, _TRAIN_EVAL_100K_CONFIGS[3][1]),
}


# Appeler eval_ddqn pour chaque nom de checkpoint (1k, 10k, 100k).
def _ddqn_eval_checkpoints_100k(EnvCls, seed: int, hidden_size: int) -> None:
    env = EnvCls()
    try:
        for n in _CHECKPOINTS_100K:
            name = f"{_MODEL_PREFIX_100K}_{n}.pkl"
            print(f"\nEVAL {EnvCls.__name__} | {name}")
            eval_ddqn(
                env,
                num_episodes=1_000,
                model_name=name,
                seed=seed,
                hidden_size=hidden_size,
            )
    finally:
        env.close()


# Entraînement 100k complet + eval sur tous les envs (seed 42 dans la boucle).
def main_train_eval_100k() -> None:
    seed = 42
    for EnvCls, train_kw in _TRAIN_EVAL_100K_CONFIGS:
        h = int(train_kw["hidden_size"])
        print(
            f"\nTRAIN DDQN {EnvCls.__name__} | "
            f"{_NUM_EPISODES_100K} episodes | checkpoints {_CHECKPOINTS_100K}"
        )
        env = EnvCls()
        try:
            ddqn(
                env,
                num_episodes=_NUM_EPISODES_100K,
                seed=seed,
                checkpoints=_CHECKPOINTS_100K,
                **train_kw,
            )
        finally:
            env.close()

        print(f"\nEVAL DDQN {EnvCls.__name__}")
        _ddqn_eval_checkpoints_100k(EnvCls, seed, h)


def _train_single_env_from_cli(env_name: str, episodes: int, seed: int) -> None:
    EnvCls, train_kw = _ENV_TRAIN_CONFIGS[env_name]
    print(f"TRAIN DDQN {EnvCls.__name__} | episodes={episodes} | seed={seed}")
    env = EnvCls()
    try:
        ddqn(
            env,
            num_episodes=episodes,
            seed=seed,
            checkpoints=_CHECKPOINTS_100K,
            **train_kw,
        )
    finally:
        env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DDQN on a selected environment.")
    parser.add_argument("--env", choices=tuple(_ENV_TRAIN_CONFIGS.keys()), required=True)
    parser.add_argument("--episodes", "-n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _train_single_env_from_cli(args.env, args.episodes, args.seed)