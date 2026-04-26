import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt

from deeprl_5iabd.envs.tictactoe import TicTacToeEnv
from deeprl_5iabd.envs.quarto import QuartoEnv, Phase


# MLP simple pour estimer les Q-values (identique à DQN)
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


def _choose_action_epsilon_greedy(
    state: np.ndarray,
    mask: np.ndarray,
    q_net: QNetwork,
    epsilon: float,
) -> int:
    available = np.where(np.asarray(mask) == 1)[0]

    # Exploration
    if np.random.random() < epsilon:
        return int(np.random.choice(available))

    # Exploitation
    x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = q_net(x)[0].numpy()

    masked_q = np.full_like(q_values, -np.inf)
    masked_q[available] = q_values[available]
    return int(np.argmax(masked_q))


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

    # Q(s, a) : prédiction du réseau ONLINE pour l'action jouée
    q_sa = q_net(x)[0, action]

    # Calcul de la cible Double DQN (sans gradient)
    with torch.no_grad():
        if done:
            target_q_next = torch.tensor(0.0)
        else:
            # (1) SÉLECTION : argmax de Q_online(s', ·) restreint aux actions légales
            q_next_online = q_net(x_next)[0].numpy()
            available_next = np.where(np.asarray(next_mask) == 1)[0]

            if len(available_next) == 0:
                target_q_next = torch.tensor(0.0)
            else:
                masked_online = np.full_like(q_next_online, -np.inf)
                masked_online[available_next] = q_next_online[available_next]
                best_next_action = int(np.argmax(masked_online))

                # (2) ÉVALUATION : on regarde Q_target(s', best_next_action)
                q_next_target = target_net(x_next)[0]
                target_q_next = q_next_target[best_next_action]

        # Cible TD Double DQN : r + gamma * Q_target(s', argmax_a Q_online(s', a))
        td_target = reward + gamma * target_q_next

    # Perte MSE entre Q_online(s, a) prédit et la cible
    loss = loss_fn(q_sa, td_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


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
) -> QNetwork:
    if q_net is None:
        q_net = QNetwork(env, hidden_size=hidden_size)

    # Target network : copie figée du réseau online
    target_net = QNetwork(env, hidden_size=hidden_size)
    target_net.load_state_dict(q_net.state_dict())
    for p in target_net.parameters():
        p.requires_grad = False

    # Compteur global de pas pour la synchro périodique du target_net
    global_step = 0

    # RMSProp avec momentum 0.95 (cf. annexe du papier DDQN)
    optimizer = optim.RMSprop(q_net.parameters(), lr=lr, momentum=0.95)
    loss_fn = nn.MSELoss()

    # Nombre d'épisodes sur lesquels ε décroît linéairement
    epsilon_anneal_episodes = max(1, int(epsilon_anneal_frac * num_episodes))

    reward_per_episode = np.zeros(num_episodes)
    loss_per_episode = np.zeros(num_episodes)
    epsilon = epsilon_start

    for epoch in range(num_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        ep_rewards = []
        ep_losses = []

        while not terminated and not truncated:
            mask = env.get_action_mask()

            if isinstance(env, QuartoEnv):
                # Phase PLACE : l'agent pose la pièce qui lui a été donnée
                if env.phase == Phase.PLACE:
                    action = _choose_action_epsilon_greedy(state, mask, q_net, epsilon)
                    new_state, reward, terminated, truncated, _ = env.step(action)
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

                # Phase SELECT : l'agent choisit la pièce à donner à l'adversaire
                if env.phase == Phase.SELECT:
                    action = _choose_action_epsilon_greedy(state, mask, q_net, epsilon)
                    new_state, reward, terminated, truncated, _ = env.step(action)

                    # Tour de l'adversaire aléatoire (PLACE + SELECT)
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
                action = _choose_action_epsilon_greedy(state, mask, q_net, epsilon)
                new_state, reward, terminated, truncated, _ = env.step(action)

                if not (terminated or truncated):
                    opp_mask = env.get_action_mask()
                    opp_action = env.action_space.sample(mask=opp_mask)
                    new_state, reward, terminated, truncated, _ = env.step(opp_action)

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
                # Environnements solo : LineWorld, GridWorld, etc.
                action = _choose_action_epsilon_greedy(state, mask, q_net, epsilon)
                new_state, reward, terminated, truncated, _ = env.step(action)
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

        # Statistiques de l'épisode
        reward_per_episode[epoch] = np.sum(ep_rewards)
        loss_per_episode[epoch] = np.mean(ep_losses) if ep_losses else 0.0

        # Décroissance LINÉAIRE d'ε de epsilon_start à epsilon_end
        # sur les epsilon_anneal_episodes premiers épisodes (cf. papier DDQN/DQN).
        frac = min(1.0, (epoch + 1) / epsilon_anneal_episodes)
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

        if epoch % 100 == 0:
            recent = reward_per_episode[max(0, epoch - 100):epoch + 1]
            wins = np.sum(recent == 1) / len(recent) * 100
            losses = np.sum(recent == -1) / len(recent) * 100
            print(
                f"Episode {epoch}: W={wins:.0f}% L={losses:.0f}% "
                f"| ε={epsilon:.3f} | Loss={loss_per_episode[epoch]:.4f}"
            )

    # Tracé des courbes 
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
    ax1.set_title(f"Double DQN - {env} | Win/Loss rate")
    ax1.legend()

    ax2.plot(mean_reward, color="blue")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Épisode")
    ax2.set_ylabel("Reward moyen (100 épisodes)")
    ax2.set_title(f"Double DQN - {env} | Mean reward")
    ax2.set_ylim(-1.05, 1.05)

    ax3.plot(loss_per_episode, label="Loss")
    ax3.set_xlabel("Épisode")
    ax3.set_ylabel("Loss")
    ax3.set_title("Loss de l'algo")
    ax3.legend()

    plt.tight_layout()
    os.makedirs("doc", exist_ok=True)
    plt.savefig(f"doc/ddqn_{env}.png")

    with open(f"doc/ddqn_{env}.pkl", "wb") as f:
        pickle.dump(q_net, f)

    env.close()

    return q_net
