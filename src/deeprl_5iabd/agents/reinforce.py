import os
import time
import torch
import pickle
import numpy as np
import torch.nn as nn
import gymnasium as gym
from torch import optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from deeprl_5iabd.agents.random_agent import RandomPlayer
from deeprl_5iabd.helper import Player, softmax_with_mask
from deeprl_5iabd.envs.line_world import LineWorldEnv
from deeprl_5iabd.envs.tictactoe import TicTacToeEnv
from deeprl_5iabd.envs.grid_world import GridWorldEnv
from deeprl_5iabd.envs.quarto import QuartoEnv, Phase
from deeprl_5iabd.config import settings

class ReinforceAgent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )  

    def forward(self, x, mask):
        x = self.network(x)
        x = softmax_with_mask(x, mask)
        return x

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_space.n),
        )

    def forward(self, x, mask):
        x = self.network(x)
        return softmax_with_mask(x, mask)

class Critic(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def reinforce(
    env: gym.Env,
    reinforce_agent: ReinforceAgent = None,
    num_episodes: int = 10_000,
    lr: float = 0.001,
    gamma: float = 0.99,
    with_baseline: bool = True,
):
    if reinforce_agent is None:
        reinforce_agent = ReinforceAgent(env)

    optimizer = optim.Adam(reinforce_agent.parameters(), lr=lr)
    reward_per_episode = np.zeros(num_episodes)
    loss_per_episode = np.zeros(num_episodes)

    for epoch in range(num_episodes):
        log_probs, rewards = [], []

        state, _ = env.reset()
        terminated = False
        truncated = False

        while not terminated and not truncated:
            mask = env.get_action_mask()
            
            # env.render()
            # time.sleep(2)

            if isinstance(env, QuartoEnv):
                # Phase PLACE : l'agent place la pièce qu'il a reçue
                if env.phase == Phase.PLACE:
                    action_probs = reinforce_agent.forward(torch.tensor(state).float(), mask)
                    probs_dist = Categorical(action_probs)
                    action = probs_dist.sample()
                    log_probs.append(probs_dist.log_prob(action))
                    
                    new_state, reward, terminated, truncated, _ = env.step(action.item())
                    state = new_state
                    rewards.append(reward)
                    
                    if terminated or truncated:
                        break

                    # Après PLACE vient SELECT
                    mask = env.get_action_mask()
                    # env.render()
                    # print("agent place")
                    # time.sleep(2)
                    

                # Phase SELECT : l'agent choisit la pièce à donner
                # (on passe ici directement après PLACE, ou si on commence en SELECT)
                if env.phase == Phase.SELECT:
                    action_probs = reinforce_agent.forward(torch.tensor(state).float(), mask)
                    probs_dist = Categorical(action_probs)
                    action = probs_dist.sample()
                    log_probs.append(probs_dist.log_prob(action))
                    
                    new_state, reward, terminated, truncated, _ = env.step(action.item())
                    state = new_state
                    rewards.append(reward)
                    
                    if terminated or truncated:
                        break
                    # env.render()
                    # print("agent select")
                    # time.sleep(2)


                    # Maintenant c'est au tour de l'adversaire (PLACE + SELECT)
                    # PLACE adverse
                    mask = env.get_action_mask()
                    adv_action = env.action_space.sample(mask=mask)
                    new_state, reward, terminated, truncated, _ = env.step(adv_action)
                    state = new_state
                    
                    # env.render()
                    # print("adversaire place")
                    # time.sleep(2)

                    if terminated or truncated:
                        rewards[-1] = reward  # patch le reward avec le résultat final
                        break

                    # SELECT adverse
                    mask = env.get_action_mask()
                    adv_action = env.action_space.sample(mask=mask)
                    new_state, reward, terminated, truncated, _ = env.step(adv_action)
                    state = new_state
                    
                    # env.render()
                    # print("adversaire select")
                    # time.sleep(2)

            elif isinstance(env, TicTacToeEnv):
                action_probs = reinforce_agent.forward(
                    torch.tensor(state).float(), mask
                )
                probs_dist = Categorical(action_probs)
                action = probs_dist.sample()

                new_state, reward, terminated, truncated, _ = env.step(action.item())
                log_probs.append(probs_dist.log_prob(action))

                if not terminated and not truncated:
                    mask = env.get_action_mask()
                    action = env.action_space.sample(mask=mask)
                    new_state, reward, terminated, truncated, _ = env.step(action)

                    rewards.append(reward)
                else:
                    rewards.append(reward)

            else:
                action_probs = reinforce_agent.forward(
                    torch.tensor(state).float(), mask
                )
                probs_dist = Categorical(action_probs)
                action = probs_dist.sample()

                new_state, reward, terminated, truncated, _ = env.step(action.item())
                log_probs.append(probs_dist.log_prob(action))
                rewards.append(reward)

            state = new_state

        ## debug
        # env.render()
        # time.sleep(5)

        ## debug
        # print(f"rewards : {sum(rewards)}")
        # print(f"reward : {reward}")

        reward_per_episode[epoch] = np.sum(rewards)

        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns).float()

        if with_baseline:
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = torch.stack([-lp * G for lp, G in zip(log_probs, returns)]).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_per_episode[epoch] = loss.item()

        if epoch % 100 == 0:
            recent = reward_per_episode[max(0, epoch-100):epoch+1]
            wins = np.sum(recent == 1) / len(recent) * 100
            losses = np.sum(recent == -1) / len(recent) * 100
            print(f"Episode {epoch}: W={wins:.0f}% L={losses:.0f}% | Loss = {loss_per_episode[epoch]:.4f}")


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    wins_rate = np.zeros(num_episodes)
    losses_rate = np.zeros(num_episodes)

    for t in range(num_episodes):
        recent = reward_per_episode[max(0, t - 100):t + 1]
        wins_rate[t] = np.sum(recent == 1) / len(recent) * 100
        losses_rate[t] = np.sum(recent == -1) / len(recent) * 100

    ax1.plot(wins_rate, label="Victoires %", color="green")
    ax1.plot(losses_rate, label="Défaites %", color="red")
    ax1.set_xlabel("Épisode")
    ax1.set_ylabel("% sur 100 épisodes")
    ax1.set_title(f"REINFORCE - {env} | Win/Loss rate")
    ax1.legend()

    ax2.plot(loss_per_episode, label="Loss")
    ax2.set_xlabel("Épisode")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss de l'algo")
    ax2.legend()

    plt.tight_layout()
    save_dir = f"{settings.training_logs_path}/reinforce_baseline_{with_baseline}/{env.unwrapped}"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/plot_reward.png")

    model_dir = f"{settings.models_path}/reinforce_baseline_{with_baseline}/{env.unwrapped}"
    os.makedirs(model_dir, exist_ok=True)
    with open(f"{model_dir}/model.pkl", "wb") as f:
        pickle.dump(reinforce_agent, f)

    env.close()

    return reinforce_agent

def reinforce_critic_baseline(
    env,
    actor: Actor,
    critic: Critic,
    num_episodes: int = 10_000,
    lr_actor: float = 0.001,
    lr_critic: float = 0.001,
    gamma: float = 0.99,
    env_name: str = "tictactoe",
):
    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)

    reward_per_episode = np.zeros(num_episodes)
    loss_actor_per_episode = np.zeros(num_episodes)
    loss_critic_per_episode = np.zeros(num_episodes)

    for epoch in range(num_episodes):
        log_probs, rewards, states = [], [], []

        state, _ = env.reset()
        terminated = False
        truncated = False

        while not terminated and not truncated:
            mask = env.get_action_mask()

            if env.current_player == env.agent_player:
                state_tensor = torch.tensor(state).float()
                action_probs = actor(state_tensor, mask)
                probs_dist = Categorical(action_probs)
                action = probs_dist.sample()

                states.append(state_tensor)
                log_probs.append(probs_dist.log_prob(action))

                new_state, reward, terminated, truncated, _ = env.step(action.item())
                rewards.append(reward)
            else:
                new_state, reward, terminated, truncated, _ = env.step(
                    env.action_space.sample(mask=mask)
                )

            state = new_state

        if not log_probs:
            continue

        reward_per_episode[epoch] = sum(rewards)

        # Returns
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns).float()

        states_tensor = torch.stack(states)
        values = critic(states_tensor)          # V(s) pour chaque état visité
        advantages = returns - values.detach()  # detach : on ne backprop pas le critic ici

        # Actor loss
        loss_actor = torch.stack([-lp * adv for lp, adv in zip(log_probs, advantages)]).sum()

        optimizer_actor.zero_grad()
        loss_actor.backward()
        optimizer_actor.step()

        # Critic loss
        loss_critic = nn.functional.mse_loss(values, returns)

        optimizer_critic.zero_grad()
        loss_critic.backward()
        optimizer_critic.step()

        loss_actor_per_episode[epoch] = loss_actor.item()
        loss_critic_per_episode[epoch] = loss_critic.item()

        if epoch % 100 == 0:
            recent = reward_per_episode[max(0, epoch - 100):epoch + 1]
            wins = np.sum(recent == 1) / len(recent) * 100
            losses = np.sum(recent == -1) / len(recent) * 100
            print(
                f"Episode {epoch}: W={wins:.0f}% L={losses:.0f}% "
                f"| Loss Actor={loss_actor_per_episode[epoch]:.4f} "
                f"| Loss Critic={loss_critic_per_episode[epoch]:.4f}"
            )

    # Plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    sum_rewards = np.array([
        np.mean(reward_per_episode[max(0, t - 100):t + 1])
        for t in range(num_episodes)
    ])
    ax1.plot(sum_rewards)
    ax1.set_xlabel("Épisode")
    ax1.set_ylabel("Reward moyen (100 épisodes)")
    ax1.set_title(f"REINFORCE + Critic Baseline - {env_name}")

    ax2.plot(loss_actor_per_episode)
    ax2.set_xlabel("Épisode")
    ax2.set_ylabel("Loss Actor")
    ax2.set_title("Loss Actor par épisode")

    ax3.plot(loss_critic_per_episode)
    ax3.set_xlabel("Épisode")
    ax3.set_ylabel("Loss Critic")
    ax3.set_title("Loss Critic par épisode")

    plt.tight_layout()
    plt.savefig(f"reinforce_critic_{env_name}.png")

    with open(f"reinforce_critic_{env_name}.pkl", "wb") as f:
        pickle.dump((actor, critic), f)

    env.close()
    return actor, critic
