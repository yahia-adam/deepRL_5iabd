import os
import pickle
import time
from typing import Optional, Tuple

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
from deeprl_5iabd.agents.ddqn_replay import ddqn_batch_q_sa_and_td_target
from deeprl_5iabd.envs.grid_world import GridWorldEnv
from deeprl_5iabd.envs.line_world import LineWorldEnv
from deeprl_5iabd.envs.quarto import QuartoEnv, Phase
from deeprl_5iabd.envs.tictactoe import TicTacToeEnv
from deeprl_5iabd.helper import plot_metric


# Segment tree binaire pour sommes de priorités (proportional sampling).
class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.empty(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    # Propager le changement de somme jusqu'à la racine après mise à jour d'une feuille.
    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # Parcourir l'arbre par masse cumulée (variable s) pour tirer une feuille.
    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    # La racine stocke la somme totale des priorités pour tous les éléments stockés.
    def total(self) -> float:
        return self.tree[0]

    # Ajouter une transition avec priorité max pour que les nouveaux tirages soient vite appris.
    def add(self, priority: float, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    # Fixer la priorité d'une feuille et recalculer les agrégats.
    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    # Tirage : renvoie l'index dans l'arbre, la priorité, le tuple de transition stocké.
    def get(self, s: float) -> Tuple[int, float, object]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


# Prioritized replay : sampling proportionnel à la priorité à la puissance alpha ; importance weights avec beta.
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, per_eps: float = 1e-6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.per_eps = per_eps
        self.max_priority = 1.0

    # Nombre de transitions dans l'arbre (remplissage partiel si capacité non atteinte).
    def __len__(self):
        return self.tree.n_entries

    # Stocker avec priorité élevée par défaut (corrigée ensuite par les TD errors).
    def push(self, s, a, r, s_prime, next_mask, done):
        data = (s, a, r, s_prime, next_mask, done)
        self.tree.add(self.max_priority ** self.alpha, data)

    # Découper la masse de priorité totale en batch_size strates ; un tirage uniforme par strate.
    def sample(self, batch_size: int, beta: float):
        batch_data = []
        tree_indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)
        total = self.tree.total()
        segment = total / batch_size
        for i in range(batch_size):
            s_rnd = np.random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s_rnd)
            batch_data.append(data)
            tree_indices[i] = idx
            priorities[i] = priority
        sampling_probs = priorities / total
        weights = (self.tree.n_entries * sampling_probs) ** (-beta)
        weights /= weights.max()
        s, a, r, s_prime, next_mask, done = zip(*batch_data)
        batch = (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(s_prime, dtype=np.float32),
            np.array(next_mask, dtype=np.float32),
            np.array(done, dtype=np.float32),
        )
        return batch, tree_indices, weights.astype(np.float32)

    # Rafraîchir les priorités depuis les dernières TD errors : (abs(TD error) + eps) à la puissance alpha.
    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray):
        priorities = (np.abs(td_errors) + self.per_eps)
        for idx, p in zip(tree_indices, priorities):
            self.tree.update(idx, float(p) ** self.alpha)
        self.max_priority = max(self.max_priority, float(priorities.max()))


# MSE pondérée des TD errors sur minibatch PER ; met à jour les priorités dans l'arbre.
def _ddqn_per_update(
    q_net: QNetwork,
    target_net: QNetwork,
    optimizer: optim.Optimizer,
    replay_buffer: PrioritizedReplayBuffer,
    batch_size: int,
    gamma: float,
    beta: float,
) -> float:
    batch, tree_indices, is_weights = replay_buffer.sample(batch_size, beta)
    states, actions, rewards, next_states, next_masks, dones = batch

    X = torch.tensor(states)
    X_next = torch.tensor(next_states)
    actions_t = torch.tensor(actions, dtype=torch.long)
    rewards_t = torch.tensor(rewards)
    dones_t = torch.tensor(dones)
    masks_t = torch.tensor(next_masks)
    is_weights_t = torch.tensor(is_weights)

    q_sa, td_target = ddqn_batch_q_sa_and_td_target(
        q_net, target_net, X, X_next, actions_t, rewards_t, dones_t, masks_t, gamma,
    )

    td_errors = td_target - q_sa
    loss = (is_weights_t * td_errors.pow(2)).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    replay_buffer.update_priorities(tree_indices, td_errors.detach().numpy())
    return loss.item()


# Push transition ; éventuellement optimiser avec un minibatch PER (beta = importance-sampling annealing).
def _store_and_train(
    replay_buffer, q_net, target_net, optimizer,
    state, action, reward, new_state, next_mask, done,
    batch_size, gamma, beta, learning_starts, train_freq,
    global_step,
) -> Optional[float]:
    replay_buffer.push(state, action, reward, new_state, next_mask, done)
    if len(replay_buffer) < max(batch_size, learning_starts):
        return None
    if global_step % train_freq != 0:
        return None
    return _ddqn_per_update(
        q_net, target_net, optimizer, replay_buffer,
        batch_size, gamma, beta,
    )


# DDQN + prioritized replay : target sync, beta anneal, hyperparamètres buffer / train frequency.
def ddqn_per(
    env: gym.Env,
    q_net: QNetwork = None,
    num_episodes: int = 10_000,
    lr: float = 6.25e-5,
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
    alpha: float = 0.6,
    beta_start: float = 0.4,
    beta_end: float = 1.0,
    per_eps: float = 1e-6,
    seed: int = 42,
    checkpoints: tuple = (1_000, 10_000, 100_000),
) -> QNetwork:
    set_seed(seed)
    if q_net is None:
        q_net = QNetwork(env, hidden_size=hidden_size)

    agent_name = "ddqn_per"
    model_dir = f"{settings.models_path}/{agent_name}/{env.unwrapped}/seed_{seed}"
    os.makedirs(model_dir, exist_ok=True)
    plot_dir = f"{settings.training_logs_dir}/{agent_name}/{env.unwrapped}/seed_{seed}"

    target_net = QNetwork(env, hidden_size=hidden_size)
    target_net.load_state_dict(q_net.state_dict())
    for p in target_net.parameters():
        p.requires_grad = False

    optimizer = optim.RMSprop(q_net.parameters(), lr=lr, momentum=0.95)
    replay_buffer = PrioritizedReplayBuffer(buffer_capacity, alpha=alpha, per_eps=per_eps)

    rewards_history = np.zeros(num_episodes)
    loss_history = np.zeros(num_episodes)
    nbr_steps_history = np.zeros(num_episodes, dtype=int)
    time_per_move_history = np.zeros(num_episodes)

    epsilon = epsilon_start
    global_step = 0
    epsilon_anneal_episodes = max(1, int(epsilon_anneal_frac * num_episodes))

    # Beta importance-sampling : croît de beta_start à beta_end sur l'entraînement.
    def get_beta(epoch: int) -> float:
        frac = epoch / max(1, num_episodes - 1)
        return beta_start + (beta_end - beta_start) * frac

    # Après un env step : push to buffer, éventuellement gradient PER + target sync.
    def maybe_update(state, action, reward, new_state, next_mask, done, beta):
        nonlocal global_step
        global_step += 1
        loss = _store_and_train(
            replay_buffer, q_net, target_net, optimizer,
            state, action, reward, new_state, next_mask, done,
            batch_size, gamma, beta, learning_starts, train_freq,
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
        beta = get_beta(epoch)
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
                        terminated or truncated, beta,
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
                        terminated or truncated, beta,
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
                    terminated or truncated, beta,
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
                    terminated or truncated, beta,
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

        frac_eps = min(1.0, (epoch + 1) / epsilon_anneal_episodes)
        epsilon = epsilon_start + frac_eps * (epsilon_end - epsilon_start)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            recent = rewards_history[max(0, epoch - 99):epoch + 1]
            recent_tpm = time_per_move_history[max(0, epoch - 99):epoch + 1]
            win_rate = float(np.mean(recent == 1) * 100)
            loss_rate = float(np.mean(recent == -1) * 100)
            print(
                f"[{agent_name} | {env.unwrapped} | seed={seed}] Episode {epoch + 1} | "
                f"Win={win_rate:.0f}% Lose={loss_rate:.0f}% | buf={len(replay_buffer)} | "
                f"Loss={loss_history[epoch]:.4f} | beta={beta:.2f} | "
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


# Charger checkpoints ddqn_per (convention de chemins) et écrire l'eval JSON.
def eval_ddqn_per(
    env: gym.Env,
    num_episodes: int = 1_000,
    model_name: str = "policy_ddqn_per_10000.pkl",
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
        algo_subdir="ddqn_per",
        num_episodes=num_episodes,
        max_episode_steps=max_episode_steps,
        log_tag="EVAL DDQN+PER",
    )


# Pilote 100k PER : buffers plus grands sur gros envs ; alpha/beta PER dans kwargs par env.
_NUM_EPISODES_100K = 100_000
_CHECKPOINTS_100K = (1_000, 10_000, 100_000)
_MODEL_PREFIX_100K = "policy_ddqn_per"

_TRAIN_EVAL_100K_CONFIGS = [
    (
        LineWorldEnv,
        dict(
            lr=6.25e-5,
            hidden_size=16,
            epsilon_anneal_frac=0.1,
            epsilon_end=0.02,
            gamma=0.9,
            buffer_capacity=5_000,
            batch_size=32,
            learning_starts=200,
            target_update_freq=200,
            train_freq=4,
            alpha=0.6,
            beta_start=0.4,
            beta_end=1.0,
        ),
    ),
    (
        GridWorldEnv,
        dict(
            lr=6.25e-5,
            hidden_size=32,
            epsilon_anneal_frac=0.3,
            epsilon_end=0.1,
            buffer_capacity=20_000,
            batch_size=32,
            learning_starts=500,
            target_update_freq=500,
            train_freq=4,
            alpha=0.6,
            beta_start=0.4,
            beta_end=1.0,
        ),
    ),
    (
        TicTacToeEnv,
        dict(
            lr=6.25e-5,
            hidden_size=128,
            epsilon_anneal_frac=0.5,
            epsilon_end=0.1,
            buffer_capacity=50_000,
            batch_size=32,
            learning_starts=1_000,
            target_update_freq=500,
            train_freq=4,
            alpha=0.6,
            beta_start=0.4,
            beta_end=1.0,
        ),
    ),
    (
        QuartoEnv,
        dict(
            lr=6.25e-5,
            hidden_size=256,
            epsilon_anneal_frac=0.6,
            epsilon_end=0.1,
            buffer_capacity=100_000,
            batch_size=32,
            learning_starts=2_000,
            target_update_freq=1_000,
            train_freq=4,
            alpha=0.6,
            beta_start=0.4,
            beta_end=1.0,
        ),
    ),
]

_ENV_TRAIN_CONFIGS = {
    "lineworld": (LineWorldEnv, _TRAIN_EVAL_100K_CONFIGS[0][1]),
    "gridworld": (GridWorldEnv, _TRAIN_EVAL_100K_CONFIGS[1][1]),
    "tictactoe": (TicTacToeEnv, _TRAIN_EVAL_100K_CONFIGS[2][1]),
    "quarto": (QuartoEnv, _TRAIN_EVAL_100K_CONFIGS[3][1]),
}


# Eval des noms de checkpoint standardisés pour un type d'env.
def _ddqn_per_eval_checkpoints_100k(EnvCls, seed: int, hidden_size: int) -> None:
    env = EnvCls()
    try:
        for n in _CHECKPOINTS_100K:
            name = f"{_MODEL_PREFIX_100K}_{n}.pkl"
            print(f"\nEVAL {EnvCls.__name__} | {name}")
            eval_ddqn_per(
                env,
                num_episodes=1_000,
                model_name=name,
                seed=seed,
                hidden_size=hidden_size,
            )
    finally:
        env.close()


# Balayage complet : chaque env dans _TRAIN_EVAL_100K_CONFIGS ; 100k puis eval aux trois checkpoints.
def main_train_eval_100k() -> None:
    seed = 42
    for EnvCls, train_kw in _TRAIN_EVAL_100K_CONFIGS:
        h = int(train_kw["hidden_size"])
        print(
            f"\nTRAIN DDQN+PER {EnvCls.__name__} | "
            f"{_NUM_EPISODES_100K} episodes | checkpoints {_CHECKPOINTS_100K}"
        )
        env = EnvCls()
        try:
            ddqn_per(
                env,
                num_episodes=_NUM_EPISODES_100K,
                seed=seed,
                checkpoints=_CHECKPOINTS_100K,
                **train_kw,
            )
        finally:
            env.close()

        print(f"\nEVAL DDQN+PER {EnvCls.__name__}")
        _ddqn_per_eval_checkpoints_100k(EnvCls, seed, h)


def _train_single_env_from_cli(env_name: str, episodes: int, seed: int) -> None:
    EnvCls, train_kw = _ENV_TRAIN_CONFIGS[env_name]
    print(f"TRAIN DDQN+PER {EnvCls.__name__} | episodes={episodes} | seed={seed}")
    env = EnvCls()
    try:
        ddqn_per(
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

    parser = argparse.ArgumentParser(description="Train DDQN+PER on a selected environment.")
    parser.add_argument("--env", choices=tuple(_ENV_TRAIN_CONFIGS.keys()), required=True)
    parser.add_argument("--episodes", "-n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _train_single_env_from_cli(args.env, args.episodes, args.seed)