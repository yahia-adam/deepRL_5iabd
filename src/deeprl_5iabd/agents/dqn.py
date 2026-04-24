import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt

from deeprl_5iabd.envs.tictactoe import TicTacToeEnv
from deeprl_5iabd.envs.quarto import QuartoEnv, Phase


# MLP simple pour estimer les Q-values
class QNetwork(nn.Module):
    def __init__(self, env: gym.Env, hidden_size: int = 128):
        super().__init__()
        # Taille d'entrée = produit des dimensions de l'espace d'observation (flatten)
        input_size = int(np.array(env.observation_space.shape).prod())
        # Taille de sortie = nombre d'actions discrètes possibles
        output_size = int(env.action_space.n)
        # Réseau : 2 couches cachées ReLU + 1 couche linéaire de sortie
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Passe avant : renvoie les Q-values pour chaque action
        return self.network(x)


def _choose_action_epsilon_greedy(
    state: np.ndarray,
    mask: np.ndarray,
    q_net: QNetwork,
    epsilon: float,
) -> int:
    """Sélection d'action ε-greedy avec masque d'actions légales."""
    # Liste des indices d'actions autorisées par le masque
    available = np.where(np.asarray(mask) == 1)[0]

    # Exploration : on tire une action au hasard parmi les actions légales
    if np.random.random() < epsilon:
        return int(np.random.choice(available))

    # Exploitation : on convertit l'état en tenseur (avec dim batch)
    x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = q_net(x)[0].numpy()

    # On met à -inf les Q-values des actions interdites pour que argmax les ignore
    masked_q = np.full_like(q_values, -np.inf)
    masked_q[available] = q_values[available]
    # On renvoie l'action légale ayant la plus grande Q-value
    return int(np.argmax(masked_q))


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
    """Mise à jour TD en ligne sur une seule transition (pas de replay buffer)."""
    # Tenseur de l'état courant (1, obs_size)
    x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    # Tenseur de l'état suivant (1, obs_size)
    x_next = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

    # Q(s, a) : Q-value prédite pour l'action effectivement jouée
    q_sa = q_net(x)[0, action]

    # Calcul de la cible TD sans propager les gradients
    with torch.no_grad():
        if done:
            # Si épisode terminé, pas d'état suivant => max Q(s', a') = 0
            max_q_next = torch.tensor(0.0)
        else:
            # Q-values du prochain état
            q_next = q_net(x_next)[0].numpy()
            # Actions encore légales dans l'état suivant
            available_next = np.where(np.asarray(next_mask) == 1)[0]
            if len(available_next) == 0:
                # Aucune action légale (edge case) => on met 0
                max_q_next = torch.tensor(0.0)
            else:
                # max_{a'} Q(s', a') restreint aux actions légales
                max_q_next = torch.tensor(float(np.max(q_next[available_next])))

        # Cible TD : r + γ · max Q(s', a')
        # (max_q_next est déjà mis à 0 plus haut si done, donc pas de masque ici)
        td_target = reward + gamma * max_q_next

    # Perte MSE entre Q(s,a) prédit et la cible TD
    loss = loss_fn(q_sa, td_target)
    # Remise à zéro des gradients accumulés
    optimizer.zero_grad()
    # Rétropropagation
    loss.backward()
    # Mise à jour des poids du réseau
    optimizer.step()
    return loss.item()


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
) -> QNetwork:
    """DQN en ligne (sans replay buffer) : une mise à jour par pas d'environnement."""
    # Création du réseau si aucun n'est fourni
    if q_net is None:
        q_net = QNetwork(env, hidden_size=hidden_size)

    # Optimiseur Adam et fonction de perte MSE (classique pour DQN)
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Tableaux de suivi : somme des récompenses et perte moyenne par épisode
    reward_per_episode = np.zeros(num_episodes)
    loss_per_episode = np.zeros(num_episodes)
    # ε initial (exploration maximale)
    epsilon = epsilon_start

    for epoch in range(num_episodes):
        # Réinitialisation de l'environnement au début de chaque épisode
        state, _ = env.reset()
        terminated = False
        truncated = False
        # Récompenses et pertes collectées sur l'épisode en cours
        ep_rewards = []
        ep_losses = []

        while not terminated and not truncated:
            # Masque des actions légales dans l'état courant
            mask = env.get_action_mask()

            if isinstance(env, QuartoEnv):
                # Phase PLACE : l'agent pose la pièce qui lui a été donnée
                if env.phase == Phase.PLACE:
                    # Choix d'action ε-greedy
                    action = _choose_action_epsilon_greedy(state, mask, q_net, epsilon)
                    # Exécution de l'action dans l'environnement
                    new_state, reward, terminated, truncated, _ = env.step(action)
                    # Masque du nouvel état pour la cible TD
                    next_mask = env.get_action_mask()
                    # Mise à jour TD en ligne
                    loss = _td_update(
                        q_net, optimizer, loss_fn,
                        state, action, reward, new_state, next_mask,
                        terminated or truncated, gamma,
                    )
                    ep_losses.append(loss)
                    ep_rewards.append(reward)
                    # Transition vers le nouvel état
                    state = new_state

                    # Si partie finie après le placement, on quitte
                    if terminated or truncated:
                        break
                    # Sinon, on met à jour le masque pour la phase SELECT
                    mask = env.get_action_mask()

                # Phase SELECT : l'agent choisit la pièce à donner à l'adversaire
                if env.phase == Phase.SELECT:
                    # Choix d'action ε-greedy pour la sélection de pièce
                    action = _choose_action_epsilon_greedy(state, mask, q_net, epsilon)
                    # Exécution (l'env passe au tour de l'adversaire)
                    new_state, reward, terminated, truncated, _ = env.step(action)

                    # Tour de l'adversaire aléatoire (PLACE puis SELECT)
                    if not (terminated or truncated):
                        # PLACE adverse
                        opp_mask = env.get_action_mask()
                        opp_action = env.action_space.sample(mask=opp_mask)
                        new_state, opp_reward, terminated, truncated, _ = env.step(opp_action)

                        if terminated or truncated:
                            # L'adversaire gagne/nulle sur son placement => on reporte le reward final
                            reward = opp_reward
                        else:
                            # SELECT adverse (donne la pièce suivante à l'agent)
                            opp_mask = env.get_action_mask()
                            opp_action = env.action_space.sample(mask=opp_mask)
                            new_state, opp_reward, terminated, truncated, _ = env.step(opp_action)
                            if terminated or truncated:
                                reward = opp_reward

                    # Masque post-adversaire et update TD avec le reward (potentiellement) patché
                    next_mask = env.get_action_mask()
                    loss = _td_update(
                        q_net, optimizer, loss_fn,
                        state, action, reward, new_state, next_mask,
                        terminated or truncated, gamma,
                    )
                    ep_losses.append(loss)
                    # On enregistre le reward APRÈS patch par opp_reward
                    # (sinon les défaites ne sont jamais comptées dans les stats)
                    ep_rewards.append(reward)
                    state = new_state

            elif isinstance(env, TicTacToeEnv):
                # Tour de l'agent
                action = _choose_action_epsilon_greedy(state, mask, q_net, epsilon)
                new_state, reward, terminated, truncated, _ = env.step(action)

                # Tour de l'adversaire aléatoire (pour que s' soit l'état APRÈS l'adversaire)
                if not (terminated or truncated):
                    opp_mask = env.get_action_mask()
                    opp_action = env.action_space.sample(mask=opp_mask)
                    new_state, reward, terminated, truncated, _ = env.step(opp_action)

                # Mise à jour TD sur la transition agent -> post-adversaire
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
                # Environnements solo : LineWorld, GridWorld, etc.
                action = _choose_action_epsilon_greedy(state, mask, q_net, epsilon)
                new_state, reward, terminated, truncated, _ = env.step(action)
                next_mask = env.get_action_mask()
                loss = _td_update(
                    q_net, optimizer, loss_fn,
                    state, action, reward, new_state, next_mask,
                    terminated or truncated, gamma,
                )
                ep_losses.append(loss)
                ep_rewards.append(reward)
                state = new_state

        # Agrégation des statistiques de l'épisode
        reward_per_episode[epoch] = np.sum(ep_rewards)
        loss_per_episode[epoch] = np.mean(ep_losses) if ep_losses else 0.0

        # Décroissance exponentielle d'ε (bornée par epsilon_end)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Log toutes les 100 épisodes : taux de victoire/défaite sur la fenêtre glissante
        if epoch % 100 == 0:
            recent = reward_per_episode[max(0, epoch - 100):epoch + 1]
            wins = np.sum(recent == 1) / len(recent) * 100
            losses = np.sum(recent == -1) / len(recent) * 100
            print(
                f"Episode {epoch}: W={wins:.0f}% L={losses:.0f}% "
                f"| ε={epsilon:.3f} | Loss={loss_per_episode[epoch]:.4f}"
            )

    # Tracé des courbes de performance (3 sous-graphiques)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # --- Subplot 1 : taux de victoires et défaites sur fenêtre glissante de 100 épisodes
    wins_rate = np.zeros(num_episodes)
    losses_rate = np.zeros(num_episodes)
    mean_reward = np.zeros(num_episodes)
    for t in range(num_episodes):
        recent = reward_per_episode[max(0, t - 100):t + 1]
        wins_rate[t] = np.sum(recent == 1) / len(recent) * 100
        losses_rate[t] = np.sum(recent == -1) / len(recent) * 100
        # Reward moyen sur la fenêtre glissante (style "reinforce_mean_baseline")
        mean_reward[t] = np.mean(recent)

    ax1.plot(wins_rate, label="Victoires %", color="green")
    ax1.plot(losses_rate, label="Défaites %", color="red")
    ax1.set_xlabel("Épisode")
    ax1.set_ylabel("% sur 100 épisodes")
    ax1.set_title(f"DQN (sans replay) - {env} | Win/Loss rate")
    ax1.legend()

    # --- Subplot 2 : reward moyen (une seule courbe synthétique, style 2)
    ax2.plot(mean_reward, color="blue")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Épisode")
    ax2.set_ylabel("Reward moyen (100 épisodes)")
    ax2.set_title(f"DQN (sans replay) - {env} | Mean reward")
    ax2.set_ylim(-1.05, 1.05)

    # --- Subplot 3 : loss
    ax3.plot(loss_per_episode, label="Loss")
    ax3.set_xlabel("Épisode")
    ax3.set_ylabel("Loss")
    ax3.set_title("Loss de l'algo")
    ax3.legend()

    plt.tight_layout()
    # Sauvegarde du graphique
    plt.savefig(f"dqn_no_replay_{env}.png")

    # Sauvegarde du modèle entraîné
    with open(f"dqn_no_replay_{env}.pkl", "wb") as f:
        pickle.dump(q_net, f)

    # Fermeture propre de l'environnement
    env.close()

    return q_net
