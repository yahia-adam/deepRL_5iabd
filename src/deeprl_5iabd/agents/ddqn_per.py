import os
import pickle
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt

from deeprl_5iabd.envs.tictactoe import TicTacToeEnv
from deeprl_5iabd.envs.quarto import QuartoEnv, Phase


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


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.empty(capacity, dtype=object)
        self.write = 0       # prochain index d'écriture (FIFO)
        self.n_entries = 0   # nombre actuel de transitions stockées

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, object]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, per_eps: float = 1e-6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.per_eps = per_eps
        # Priorité maximale vue jusqu'ici (donnée aux nouvelles transitions
        # pour qu'elles soient sûres d'être vues au moins une fois)
        self.max_priority = 1.0

    def __len__(self):
        return self.tree.n_entries

    def push(self, s, a, r, s_prime, next_mask, done):
        data = (s, a, r, s_prime, next_mask, done)
        # Priorité initiale = priorité max du buffer (transition "intéressante" par défaut)
        self.tree.add(self.max_priority ** self.alpha, data)

    def sample(self, batch_size: int, beta: float):
        batch_data = []
        tree_indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)

        # Découpage du segment [0, total] en `batch_size` strates équiprobables
        # (stratified sampling : améliore la diversité de l'échantillon)
        total = self.tree.total()
        segment = total / batch_size

        for i in range(batch_size):
            a_seg = segment * i
            b_seg = segment * (i + 1)
            s = np.random.uniform(a_seg, b_seg)
            idx, priority, data = self.tree.get(s)
            batch_data.append(data)
            tree_indices[i] = idx
            priorities[i] = priority

        # Probabilités de chaque transition échantillonnée
        sampling_probs = priorities / total
        # Poids IS : w_i = (N · P(i))^(-bêta), normalisés par max(w) pour stabilité
        weights = (self.tree.n_entries * sampling_probs) ** (-beta)
        weights /= weights.max()

        # Désempaquetage du batch en arrays
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

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray):
        # Priorités = (|δ| + ε)^α, on garde aussi la max pour les nouvelles transitions
        priorities = (np.abs(td_errors) + self.per_eps)
        for idx, p in zip(tree_indices, priorities):
            self.tree.update(idx, float(p) ** self.alpha)
        # Mémorise la priorité brute (avant exponent α) maximale jamais vue
        self.max_priority = max(self.max_priority, float(priorities.max()))


def _choose_action_epsilon_greedy(state, mask, q_net, epsilon):
    available = np.where(np.asarray(mask) == 1)[0]
    if np.random.random() < epsilon:
        return int(np.random.choice(available))
    x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = q_net(x)[0].numpy()
    masked_q = np.full_like(q_values, -np.inf)
    masked_q[available] = q_values[available]
    return int(np.argmax(masked_q))


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

    # Q_online(s, a)
    q_current = q_net(X)
    q_sa = q_current.gather(1, actions_t.unsqueeze(1)).squeeze(1)

    # Cible Double DQN
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

    # TD errors par transition (pour mise à jour des priorités)
    td_errors = td_target - q_sa

    # Loss MSE pondérée par les poids d'importance sampling
    # (équivalent à MSELoss(reduction='mean') mais avec poids)
    loss = (is_weights_t * td_errors.pow(2)).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Mise à jour des priorités avec les nouvelles |TD error|
    replay_buffer.update_priorities(tree_indices, td_errors.detach().numpy())

    return loss.item()


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
) -> QNetwork:
    if q_net is None:
        q_net = QNetwork(env, hidden_size=hidden_size)

    target_net = QNetwork(env, hidden_size=hidden_size)
    target_net.load_state_dict(q_net.state_dict())
    for p in target_net.parameters():
        p.requires_grad = False

    # RMSProp avec momentum 0.95 
    optimizer = optim.RMSprop(q_net.parameters(), lr=lr, momentum=0.95)
    replay_buffer = PrioritizedReplayBuffer(buffer_capacity, alpha=alpha, per_eps=per_eps)

    reward_per_episode = np.zeros(num_episodes)
    loss_per_episode = np.zeros(num_episodes)
    epsilon = epsilon_start
    global_step = 0

    # Nombre d'épisodes sur lesquels epsilon décroît linéairement
    epsilon_anneal_episodes = max(1, int(epsilon_anneal_frac * num_episodes))

    def get_beta(epoch: int) -> float:
        frac = epoch / max(1, num_episodes - 1)
        return beta_start + (beta_end - beta_start) * frac

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
        ep_rewards = []
        ep_losses = []
        beta = get_beta(epoch)

        while not terminated and not truncated:
            mask = env.get_action_mask()

            if isinstance(env, QuartoEnv):
                # Phase PLACE
                if env.phase == Phase.PLACE:
                    action = _choose_action_epsilon_greedy(state, mask, q_net, epsilon)
                    new_state, reward, terminated, truncated, _ = env.step(action)
                    next_mask = env.get_action_mask()
                    loss = maybe_update(state, action, reward, new_state, next_mask,
                                        terminated or truncated, beta)
                    if loss is not None:
                        ep_losses.append(loss)
                    ep_rewards.append(reward)
                    state = new_state

                    if terminated or truncated:
                        break
                    mask = env.get_action_mask()

                # Phase SELECT
                if env.phase == Phase.SELECT:
                    action = _choose_action_epsilon_greedy(state, mask, q_net, epsilon)
                    new_state, reward, terminated, truncated, _ = env.step(action)

                    if not (terminated or truncated):
                        opp_mask = env.get_action_mask()
                        opp_action = env.action_space.sample(mask=opp_mask)
                        new_state, opp_reward, terminated, truncated, _ = env.step(opp_action)

                        if terminated or truncated:
                            reward = opp_reward
                        else:
                            opp_mask = env.get_action_mask()
                            opp_action = env.action_space.sample(mask=opp_mask)
                            new_state, opp_reward, terminated, truncated, _ = env.step(opp_action)
                            if terminated or truncated:
                                reward = opp_reward

                    next_mask = env.get_action_mask()
                    loss = maybe_update(state, action, reward, new_state, next_mask,
                                        terminated or truncated, beta)
                    if loss is not None:
                        ep_losses.append(loss)
                    ep_rewards.append(reward)
                    state = new_state

            elif isinstance(env, TicTacToeEnv):
                action = _choose_action_epsilon_greedy(state, mask, q_net, epsilon)
                new_state, reward, terminated, truncated, _ = env.step(action)

                if not (terminated or truncated):
                    opp_mask = env.get_action_mask()
                    opp_action = env.action_space.sample(mask=opp_mask)
                    new_state, reward, terminated, truncated, _ = env.step(opp_action)

                next_mask = env.get_action_mask()
                loss = maybe_update(state, action, reward, new_state, next_mask,
                                    terminated or truncated, beta)
                if loss is not None:
                    ep_losses.append(loss)
                ep_rewards.append(reward)
                state = new_state

            else:
                action = _choose_action_epsilon_greedy(state, mask, q_net, epsilon)
                new_state, reward, terminated, truncated, _ = env.step(action)
                next_mask = env.get_action_mask()
                loss = maybe_update(state, action, reward, new_state, next_mask,
                                    terminated or truncated, beta)
                if loss is not None:
                    ep_losses.append(loss)
                ep_rewards.append(reward)
                state = new_state

        reward_per_episode[epoch] = np.sum(ep_rewards)
        loss_per_episode[epoch] = np.mean(ep_losses) if ep_losses else 0.0

        # Décroissance LINÉAIRE d'epsilon(conforme au papier DQN/DDQN)
        frac = min(1.0, (epoch + 1) / epsilon_anneal_episodes)
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

        if epoch % 100 == 0:
            recent = reward_per_episode[max(0, epoch - 100):epoch + 1]
            wins = np.sum(recent == 1) / len(recent) * 100
            losses = np.sum(recent == -1) / len(recent) * 100
            print(
                f"Ep {epoch}: W={wins:.0f}% L={losses:.0f}% "
                f"| ε={epsilon:.3f} | bêta={beta:.2f} | buf={len(replay_buffer)} "
                f"| Loss={loss_per_episode[epoch]:.4f}"
            )

    # bêta Tracé des courbes
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    wins_rate = np.zeros(num_episodes)
    losses_rate = np.zeros(num_episodes)
    mean_reward = np.zeros(num_episodes)
    for t in range(num_episodes):
        recent = reward_per_episode[max(0, t - 100):t + 1]
        wins_rate[t] = np.sum(recent == 1) / len(recent) * 100
        losses_rate[t] = np.sum(recent == -1) / len(recent) * 100
        mean_reward[t] = np.mean(recent)

    ax1.plot(wins_rate, label="Victoires %", color="green")
    ax1.plot(losses_rate, label="Défaites %", color="red")
    ax1.set_xlabel("Épisode")
    ax1.set_ylabel("% sur 100 épisodes")
    ax1.set_title(f"Double DQN + PER - {env} | Win/Loss rate")
    ax1.legend()

    ax2.plot(mean_reward, color="blue")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Épisode")
    ax2.set_ylabel("Reward moyen (100 épisodes)")
    ax2.set_title(f"Double DQN + PER - {env} | Mean reward")
    ax2.set_ylim(-1.05, 1.05)

    ax3.plot(loss_per_episode, label="Loss")
    ax3.set_xlabel("Épisode")
    ax3.set_ylabel("Loss")
    ax3.set_title("Loss de l'algo")
    ax3.legend()

    plt.tight_layout()
    os.makedirs("doc", exist_ok=True)
    plt.savefig(f"doc/ddqn_per_{env}.png")

    with open(f"doc/ddqn_per_{env}.pkl", "wb") as f:
        pickle.dump(q_net, f)

    env.close()

    return q_net
