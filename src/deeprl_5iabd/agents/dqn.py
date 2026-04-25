"""DQN online minimal.

Aucune amélioration au-delà de Q-learning : pas de replay buffer, pas de
target network, pas de double DQN. On remplace seulement la Q-table par un
réseau Q(s, a) et on applique la mise à jour TD à chaque transition.

    Q(s, a) ← Q(s, a) + α · ( r + γ max_a' Q(s', a')  −  Q(s, a) )
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _select_action(q_net, obs, mask, epsilon, device):
    valid = np.flatnonzero(mask)
    if np.random.rand() < epsilon:
        return int(np.random.choice(valid))
    with torch.no_grad():
        q = q_net(torch.as_tensor(obs, dtype=torch.float32, device=device)).cpu().numpy()
    q[mask == 0] = -np.inf
    return int(np.argmax(q))


def _opponent_step(env):
    """Adversaire random (cohérent avec ton tabulaire Q-learning)."""
    mask = env.get_action_mask()
    action = int(np.random.choice(np.flatnonzero(mask)))
    return env.step(action)


def train_dqn(
    env,
    n_episodes: int = 10_000,
    lr: float = 1e-3,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: float = 0.9995,
    device: str = "cpu",
    log_every: int = 100,
):
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_net = QNetwork(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    epsilon = eps_start
    history = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        # /!\ les envs réutilisent le même buffer numpy → toujours copier
        obs = obs.copy()
        ep_reward = 0.0
        done = False

        # Cas où l'adversaire commence
        if env.is_multi_player and env.current_player != env.agent_player:
            obs, r, term, trunc, _ = _opponent_step(env)
            obs = obs.copy()
            ep_reward += r
            done = term or trunc

        while not done:
            mask = env.get_action_mask().copy()
            action = _select_action(q_net, obs, mask, epsilon, device)

            # Coup de l'agent
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            # Coup de l'adversaire (s'il reste)
            if not done and env.is_multi_player and env.current_player != env.agent_player:
                next_obs, r_opp, term, trunc, _ = _opponent_step(env)
                reward += r_opp
                done = term or trunc

            next_obs = next_obs.copy()
            ep_reward += reward

            # ----- mise à jour TD ------------------------------------------
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            q_pred = q_net(obs_t)[action]

            with torch.no_grad():
                if done:
                    target = torch.tensor(reward, dtype=torch.float32, device=device)
                else:
                    next_mask = env.get_action_mask()
                    next_q = q_net(torch.as_tensor(next_obs, dtype=torch.float32, device=device))
                    next_q = next_q.masked_fill(
                        torch.as_tensor(next_mask == 0, device=device), float("-inf")
                    )
                    target = reward + gamma * next_q.max()

            loss = loss_fn(q_pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ----------------------------------------------------------------

            obs = next_obs

        epsilon = max(eps_end, epsilon * eps_decay)
        history.append(ep_reward)

        if (ep + 1) % log_every == 0:
            avg = float(np.mean(history[-log_every:]))
            print(f"Episode {ep+1:>6} | reward moy: {avg:+.3f} | eps: {epsilon:.3f}")

    return q_net, history


if __name__ == "__main__":
    # Exemple
    from deeprl_5iabd.envs.line_world import LineWorldEnv
    env = LineWorldEnv()
    train_dqn(env, n_episodes=2000)