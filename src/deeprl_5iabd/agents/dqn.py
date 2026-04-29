import json
import os
import pickle
import random
import re
import time
from typing import Type

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from deeprl_5iabd.config import settings
from deeprl_5iabd.envs.grid_world import GridWorldEnv
from deeprl_5iabd.envs.line_world import LineWorldEnv
from deeprl_5iabd.envs.quarto import QuartoEnv, Phase
from deeprl_5iabd.envs.tictactoe import TicTacToeEnv
from deeprl_5iabd.helper import plot_metric, plot_trace


# Fixer le hasard : Python, NumPy et Torch (CPU et CUDA).
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Un coup légal aléatoire pour l'autre joueur (environnements multijoueurs).
def opponent_step(env: gym.Env):
    mask = env.get_action_mask()
    action = env.action_space.sample(mask=mask)
    return env.step(action)


# Plafond de pas imposé dans les algos Q (pas via TimeLimit Gym) pour line/grid.
LINE_GRID_MAX_EPISODE_STEPS = 100


# Au max de pas Line/Grid : tronquer l'épisode et remettre la récompense à 0 si pas terminé.
def apply_line_grid_step_cap(
    env: gym.Env,
    n_step: int,
    reward: float,
    terminated: bool,
    truncated: bool,
) -> tuple[float, bool, bool]:
    if not isinstance(env.unwrapped, (LineWorldEnv, GridWorldEnv)):
        return reward, terminated, truncated
    if terminated:
        return reward, terminated, truncated
    if n_step >= LINE_GRID_MAX_EPISODE_STEPS:
        return 0.0, False, True
    return reward, terminated, truncated


# Aligner le max de steps d'eval sur le plafond line/grid pour coller à l'horizon d'entraînement.
def capped_eval_max_episode_steps(env: gym.Env, max_episode_steps: int) -> int:
    if isinstance(env.unwrapped, (LineWorldEnv, GridWorldEnv)):
        return min(max_episode_steps, LINE_GRID_MAX_EPISODE_STEPS)
    return max_episode_steps


# Greedy action sur coups légaux via Q-network (eval, sans exploration).
def choose_greedy_action(
    state: np.ndarray, mask: np.ndarray, q_net: torch.nn.Module,
) -> int:
    available = np.where(np.asarray(mask) == 1)[0]
    x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = q_net(x)[0].numpy()
    masked_q = np.full_like(q_values, -np.inf)
    masked_q[available] = q_values[available]
    return int(np.argmax(masked_q))


# Exécuter num_episodes greedy rollouts ; renvoie récompenses et nombres de pas par épisode.
def greedy_eval_collect(
    env: gym.Env,
    q_net: torch.nn.Module,
    num_episodes: int,
    max_episode_steps: int,
    log_tag: str,
    log_every: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    rewards_history = np.zeros(num_episodes)
    n_steps_history = np.zeros(num_episodes, dtype=int)
    is_multi = getattr(env, "is_multi_player", False)

    print(
        f"[{log_tag}] {env.unwrapped} | {num_episodes} episodes "
        f"(max {max_episode_steps} steps/ep, log every {log_every})...",
        flush=True,
    )

    for i in range(num_episodes):
        n_step = 0
        with torch.no_grad():
            state, _ = env.reset()
            done = False
            truncated = False
            reward = 0.0
            while not (done or truncated) and n_step < max_episode_steps:
                if is_multi:
                    while (
                        not (done or truncated)
                        and n_step < max_episode_steps
                        and env.current_player != env.agent_player
                    ):
                        state, reward, done, truncated, _ = opponent_step(env)
                        n_step += 1
                if done or truncated or n_step >= max_episode_steps:
                    break
                while (
                    not (done or truncated)
                    and n_step < max_episode_steps
                    and (not is_multi or env.current_player == env.agent_player)
                ):
                    action_mask = env.get_action_mask()
                    action = choose_greedy_action(state, action_mask, q_net)
                    state, reward, done, truncated, _ = env.step(action)
                    n_step += 1
        rewards_history[i] = reward
        n_steps_history[i] = n_step
        if (i + 1) % log_every == 0:
            print(f"[{log_tag}]   ... {i + 1}/{num_episodes} episodes", flush=True)

    return rewards_history, n_steps_history


# Construire le dictionnaire d'eval, écrire le JSON sous training_logs, tracer le PNG des récompenses.
def write_eval_results(
    *,
    env: gym.Env,
    model_name: str,
    seed: int,
    rewards_history: np.ndarray,
    n_steps_history: np.ndarray,
    algo_subdir: str,
    log_tag: str,
) -> dict:
    base = model_name.replace(".pkl", "")
    # Recherche du nom de l'agent et du checkpoint dans le nom du modèle
    mo = re.match(r"^policy_(.+)_(\d+)$", base)
    if mo:
        agent_name, checkpoint_str = mo.group(1), mo.group(2)
    else:
        raise ValueError(
            "Invalid model_name format for evaluation. "
            f"Expected 'policy_<agent_name>_<checkpoint>.pkl', got {model_name!r}"
        )
    checkpoint = int(checkpoint_str)

    results = {
        "env": str(env.unwrapped),
        "agent": agent_name,
        "checkpoint": checkpoint,
        "seed": seed,
        "num_episodes": len(rewards_history),
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

    json_dir = f"{settings.training_logs_dir}/{algo_subdir}/{env.unwrapped}/seed_{seed}/eval"
    os.makedirs(json_dir, exist_ok=True)
    json_path = f"{json_dir}/{agent_name}_{checkpoint}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    trace_name = f"{model_name.rsplit('.', 1)[0]}_trace.png"
    plot_trace(
        rewards_history, model_name,
        save_path=os.path.join(json_dir, trace_name),
    )

    s = results["summary"]
    print(
        f"[{log_tag} {env.unwrapped}/{agent_name}@{checkpoint} | seed={seed}] "
        f"win={s['win_rate']:.2%} loss={s['loss_rate']:.2%} draw={s['draw_rate']:.2%} "
        f"mean_steps={s['mean_steps']:.2f} -> {json_path}",
        flush=True,
    )
    return results


# Charger le checkpoint sur disque, greedy_eval_collect, write_eval_results.
def load_qnet_and_eval(
    *,
    env: gym.Env,
    q_network_class: Type[torch.nn.Module],
    model_name: str,
    seed: int,
    hidden_size: int,
    algo_subdir: str,
    num_episodes: int = 1_000,
    max_episode_steps: int = 10_000,
    log_tag: str = "eval",
) -> dict:
    set_seed(seed)
    q_net = q_network_class(env, hidden_size=hidden_size)
    model_path = f"{settings.models_path}/{algo_subdir}/{env.unwrapped}/seed_{seed}/{model_name}"
    with open(model_path, "rb") as f:
        state_dict = pickle.load(f)
    q_net.load_state_dict(state_dict)
    q_net.eval()

    rh, sh = greedy_eval_collect(
        env, q_net, num_episodes, max_episode_steps, log_tag=log_tag,
    )
    return write_eval_results(
        env=env,
        model_name=model_name,
        seed=seed,
        rewards_history=rh,
        n_steps_history=sh,
        algo_subdir=algo_subdir,
        log_tag=log_tag,
    )


# MLP : vecteur d'état vers une valeur Q par action discrète.
class QNetwork(nn.Module):
    def __init__(self, env: gym.Env, hidden_size: int = 128):
        super().__init__()
        input_size = int(np.array(env.observation_space.shape).prod())
        output_size = int(env.action_space.n)
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# Coup légal aléatoire avec probabilité epsilon, sinon greedy action selon Q (phase entraînement).
def choose_action_epsilon_greedy(
    state: np.ndarray,
    mask: np.ndarray,
    q_net: nn.Module,
    epsilon: float,
) -> int:
    available = np.where(np.asarray(mask) == 1)[0]
    if np.random.random() < epsilon:
        return int(np.random.choice(available))
    x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = q_net(x)[0].numpy()
    masked_q = np.full_like(q_values, -np.inf)
    masked_q[available] = q_values[available]
    return int(np.argmax(masked_q))


# TD step : max sur Q(next state) avec le même online network (DQN vanilla).
def _td_update(
    q_net: QNetwork,
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
            max_q_next = torch.tensor(0.0)
        else:
            q_next = q_net(x_next)[0].numpy()
            available_next = np.where(np.asarray(next_mask) == 1)[0]
           
            if len(available_next) == 0:
                max_q_next = torch.tensor(0.0)
            else:
                max_q_next = torch.tensor(float(np.max(q_next[available_next])))
        td_target = reward + gamma * max_q_next
    loss = loss_fn(q_sa, td_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# Entraîner le réseau Q en ligne avec epsilon decay ; checkpoints et courbes d'apprentissage.
def dqn(
    env: gym.Env,
    q_net: QNetwork = None,
    num_episodes: int = 10_000,
    lr: float = 1e-3,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.9995,
    hidden_size: int = 128,
    seed: int = 42,
    checkpoints: tuple = (1_000, 10_000, 100_000),
) -> QNetwork:
    set_seed(seed)
    if q_net is None:
        q_net = QNetwork(env, hidden_size=hidden_size)
    agent_name = "dqn"
    model_dir = f"{settings.models_path}/dqn/{env.unwrapped}/seed_{seed}"
    os.makedirs(model_dir, exist_ok=True)
    plot_dir = f"{settings.training_logs_dir}/dqn/{env.unwrapped}/seed_{seed}"

    optimizer = optim.RMSprop(q_net.parameters(), lr=lr, momentum=0.95)
    loss_fn = nn.MSELoss()

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
                    loss = _td_update(
                        q_net, optimizer, loss_fn,
                        state, action, reward, new_state, next_mask,
                        terminated or truncated, gamma,
                    )
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
                    loss = _td_update(
                        q_net, optimizer, loss_fn,
                        state, action, reward, new_state, next_mask,
                        terminated or truncated, gamma,
                    )
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
                loss = _td_update(
                    q_net, optimizer, loss_fn,
                    state, action, reward, new_state, next_mask,
                    terminated or truncated, gamma,
                )
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
                loss = _td_update(
                    q_net, optimizer, loss_fn,
                    state, action, reward, new_state, next_mask,
                    terminated or truncated, gamma,
                )
                ep_losses.append(loss)
                ep_rewards.append(reward)
                state = new_state

        episode_time = time.perf_counter() - episode_start
        time_per_move_history[epoch] = episode_time / max(n_step, 1)
        nbr_steps_history[epoch] = n_step
        rewards_history[epoch] = float(np.sum(ep_rewards))
        loss_history[epoch] = float(np.mean(ep_losses) if ep_losses else 0.0)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

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
            with open(f"{model_dir}/policy_dqn_{epoch + 1}.pkl", "wb") as f:
                pickle.dump(q_net.state_dict(), f)
            print(f"Model saved: {model_dir}/policy_dqn_{epoch + 1}.pkl")

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
    with open(f"{model_dir}/policy_dqn_{num_episodes}.pkl", "wb") as f:
        pickle.dump(q_net.state_dict(), f)
    env.close()
    return q_net


# Charger les poids DQN et lancer l'eval JSON standard pour cet env et ce seed.
def eval_dqn(
    env: gym.Env,
    num_episodes: int = 1_000,
    model_name: str = "policy_dqn_10000.pkl",
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
        algo_subdir="dqn",
        num_episodes=num_episodes,
        max_episode_steps=max_episode_steps,
        log_tag="EVAL DQN",
    )


# Constantes du pilote long : budget d'épisodes et pas de checkpoints pour les scripts d'eval.
_NUM_EPISODES_100K = 100_000
_CHECKPOINTS_100K = (1_000, 10_000, 100_000)

# Hyperparamètres par env : entraînement 100k puis eval des checkpoints.
_TRAIN_EVAL_100K_CONFIGS = [
    (
        LineWorldEnv,
        dict(
            lr=2.5e-4,
            hidden_size=16,
            epsilon_decay=0.995,
            epsilon_end=0.02,
            gamma=0.9,
        ),
    ),
    (
        GridWorldEnv,
        dict(
            lr=2.5e-4,
            hidden_size=32,
            epsilon_decay=0.9995,
            epsilon_end=0.1,
        ),
    ),
    (
        TicTacToeEnv,
        dict(
            lr=2.5e-4,
            hidden_size=128,
            epsilon_decay=0.9998,
        ),
    ),
    (
        QuartoEnv,
        dict(
            lr=2.5e-4,
            hidden_size=256,
            epsilon_decay=0.9999,
        ),
    ),
]

_ENV_TRAIN_CONFIGS = {
    "lineworld": (LineWorldEnv, _TRAIN_EVAL_100K_CONFIGS[0][1]),
    "gridworld": (GridWorldEnv, _TRAIN_EVAL_100K_CONFIGS[1][1]),
    "tictactoe": (TicTacToeEnv, _TRAIN_EVAL_100K_CONFIGS[2][1]),
    "quarto": (QuartoEnv, _TRAIN_EVAL_100K_CONFIGS[3][1]),
}


# Eval policy à chaque checkpoint fixe (1k, 10k, 100k) pour une classe d'env.
def _dqn_eval_checkpoints_100k(EnvCls, seed: int, hidden_size: int) -> None:
    env = EnvCls()
    try:
        for n in _CHECKPOINTS_100K:
            print(f"\nEVAL {EnvCls.__name__} | policy_dqn_{n}.pkl")
            eval_dqn(
                env,
                num_episodes=1_000,
                model_name=f"policy_dqn_{n}.pkl",
                seed=seed,
                hidden_size=hidden_size,
            )
    finally:
        env.close()


# Entraîner 100k épisodes puis évaluer tous les checkpoints pour chaque env (seed 42).
def main_train_eval_100k() -> None:
    seed = 42
    for EnvCls, train_kw in _TRAIN_EVAL_100K_CONFIGS:
        h = int(train_kw["hidden_size"])
        print(
            f"\nTRAIN DQN {EnvCls.__name__} | "
            f"{_NUM_EPISODES_100K} episodes | checkpoints {_CHECKPOINTS_100K}"
        )
        env = EnvCls()
        try:
            dqn(
                env,
                num_episodes=_NUM_EPISODES_100K,
                seed=seed,
                checkpoints=_CHECKPOINTS_100K,
                **train_kw,
            )
        finally:
            env.close()

        print(f"\nEVAL DQN {EnvCls.__name__}")
        _dqn_eval_checkpoints_100k(EnvCls, seed, h)


def _train_single_env_from_cli(env_name: str, episodes: int, seed: int) -> None:
    EnvCls, train_kw = _ENV_TRAIN_CONFIGS[env_name]
    print(f"TRAIN DQN {EnvCls.__name__} | episodes={episodes} | seed={seed}")
    env = EnvCls()
    try:
        dqn(
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

    parser = argparse.ArgumentParser(description="Train DQN on a selected environment.")
    parser.add_argument("--env", choices=tuple(_ENV_TRAIN_CONFIGS.keys()), required=True)
    parser.add_argument("--episodes", "-n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _train_single_env_from_cli(args.env, args.episodes, args.seed)