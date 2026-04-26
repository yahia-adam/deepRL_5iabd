import os
import pickle
from collections import deque
from typing import Optional

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


def _choose_action_epsilon_greedy(
    state: np.ndarray,
    mask: np.ndarray,
    q_net: QNetwork,
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

    X = torch.tensor(states)                                  # (B, obs)
    X_next = torch.tensor(next_states)                        # (B, obs)
    actions_t = torch.tensor(actions, dtype=torch.long)       # (B,)
    rewards_t = torch.tensor(rewards)                         # (B,)
    dones_t = torch.tensor(dones)                             # (B,)
    masks_t = torch.tensor(next_masks)                        # (B, n_actions)

    # Q_online(s, a) pour les actions jouées (réseau qui apprend)
    q_current = q_net(X)                                      # (B, n_actions)
    q_sa = q_current.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # (B,)

    # Cible Double DQN (pas de gradient)
    with torch.no_grad():
        # (1) SÉLECTION : argmax de Q_online(s', ·) restreint aux actions légales
        q_next_online = q_net(X_next)                          # (B, n_actions)
        # On met -inf sur les actions interdites pour que argmax les ignore
        neg_inf = torch.full_like(q_next_online, float("-inf"))
        q_next_online_masked = torch.where(masks_t > 0, q_next_online, neg_inf)
        best_next_actions = q_next_online_masked.argmax(dim=1)  # (B,)

        # (2) ÉVALUATION : Q_target(s', best_next_action)
        q_next_target = target_net(X_next)                     # (B, n_actions)
        target_q_next = q_next_target.gather(
            1, best_next_actions.unsqueeze(1)
        ).squeeze(1)                                           # (B,)

        # Si done → pas de bootstrap : on annule target_q_next
        td_target = rewards_t + gamma * target_q_next * (1.0 - dones_t)

    # Perte MSE sur le batch
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

    # Pas d'update tant que le buffer n'a pas assez d'expériences
    if len(replay_buffer) < max(batch_size, learning_starts):
        return None
    # Update seulement tous les train_freq pas
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
) -> QNetwork:
    if q_net is None:
        q_net = QNetwork(env, hidden_size=hidden_size)

    # Target network : copie figée du réseau online
    target_net = QNetwork(env, hidden_size=hidden_size)
    target_net.load_state_dict(q_net.state_dict())
    for p in target_net.parameters():
        p.requires_grad = False

    # RMSProp avec momentum 0.95 
    optimizer = optim.RMSprop(q_net.parameters(), lr=lr, momentum=0.95)
    loss_fn = nn.MSELoss()
    replay_buffer = ReplayBuffer(buffer_capacity)

    reward_per_episode = np.zeros(num_episodes)
    loss_per_episode = np.zeros(num_episodes)
    epsilon = epsilon_start
    global_step = 0

    # Nombre d'épisodes sur lesquels epsilon décroît linéairement
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
        # Synchro périodique du target_net
        if global_step % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())
        return loss

    for epoch in range(num_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        ep_rewards = []
        ep_losses = []

        while not terminated and not truncated:
            mask = env.get_action_mask()

            if isinstance(env, QuartoEnv):
                # Phase PLACE
                if env.phase == Phase.PLACE:
                    action = _choose_action_epsilon_greedy(state, mask, q_net, epsilon)
                    new_state, reward, terminated, truncated, _ = env.step(action)
                    next_mask = env.get_action_mask()
                    loss = maybe_update(state, action, reward, new_state, next_mask,
                                        terminated or truncated)
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

                    # Tour de l'adversaire (PLACE + SELECT)
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
                                        terminated or truncated)
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
                                    terminated or truncated)
                if loss is not None:
                    ep_losses.append(loss)
                ep_rewards.append(reward)
                state = new_state

            else:
                action = _choose_action_epsilon_greedy(state, mask, q_net, epsilon)
                new_state, reward, terminated, truncated, _ = env.step(action)
                next_mask = env.get_action_mask()
                loss = maybe_update(state, action, reward, new_state, next_mask,
                                    terminated or truncated)
                if loss is not None:
                    ep_losses.append(loss)
                ep_rewards.append(reward)
                state = new_state

        reward_per_episode[epoch] = np.sum(ep_rewards)
        loss_per_episode[epoch] = np.mean(ep_losses) if ep_losses else 0.0

        # Décroissance linéaire d'epsilon
        frac = min(1.0, (epoch + 1) / epsilon_anneal_episodes)
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

        if epoch % 100 == 0:
            recent = reward_per_episode[max(0, epoch - 100):epoch + 1]
            wins = np.sum(recent == 1) / len(recent) * 100
            losses = np.sum(recent == -1) / len(recent) * 100
            print(
                f"Episode {epoch}: W={wins:.0f}% L={losses:.0f}% "
                f"| epsilon={epsilon:.3f} | buf={len(replay_buffer)} "
                f"| Loss={loss_per_episode[epoch]:.4f}"
            )

    #  Tracé des courbes
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
    ax1.set_title(f"Double DQN + Replay - {env} | Win/Loss rate")
    ax1.legend()

    ax2.plot(mean_reward, color="blue")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Épisode")
    ax2.set_ylabel("Reward moyen (100 épisodes)")
    ax2.set_title(f"Double DQN + Replay - {env} | Mean reward")
    ax2.set_ylim(-1.05, 1.05)

    ax3.plot(loss_per_episode, label="Loss")
    ax3.set_xlabel("Épisode")
    ax3.set_ylabel("Loss")
    ax3.set_title("Loss de l'algo")
    ax3.legend()

    plt.tight_layout()
    os.makedirs("doc", exist_ok=True)
    plt.savefig(f"doc/ddqn_replay_{env}.png")

    with open(f"doc/ddqn_replay_{env}.pkl", "wb") as f:
        pickle.dump(q_net, f)

    env.close()

    return q_net
