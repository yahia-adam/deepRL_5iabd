import time
import os
import numpy as np
from gymnasium import Env
import matplotlib.pyplot as plt
from collections import deque

from deeprl_5iabd.envs.line_world import LineWorldEnv
from deeprl_5iabd.envs.grid_world import GridWorldEnv
from deeprl_5iabd.envs.tictactoe import TicTacToeEnv
from deeprl_5iabd.envs.quarto import QuartoEnv
from deeprl_5iabd.config import settings


def monte_carlo_random_rollout(env: Env, num_rollouts: int):

    mask = env.get_action_mask()
    action_mean_rewards = np.full(len(mask), -np.inf)

    valid_actions = np.where(mask == 1)[0]
    action_mean_rewards[valid_actions] = 0

    if len(valid_actions) == 0:
        return
    
    a_resource = num_rollouts // len(valid_actions)

    for test_action in valid_actions:
        for _ in range(a_resource):
            new_env = env.determinize()

            total_reward = 0

            _, reward, terminated, truncated, _ = new_env.step(test_action)
            total_reward += reward

            while not (terminated or truncated):
                mask = new_env.get_action_mask()
                a = new_env.action_space.sample(mask=mask)

                _, reward, terminated, truncated, _ = new_env.step(a)
                total_reward += reward

            action_mean_rewards[test_action] += total_reward

        action_mean_rewards[test_action] /= a_resource

    best_action_idx = np.argmax(action_mean_rewards)
    return int(best_action_idx)



def run_monte_carlo(env: Env, num_episodes: int, num_rollouts: int):
    rewards = []
    wins_rate, draws_rate, losses_rate = [], [], []

    for episode in range(num_episodes):
        done = False
        env.reset()
        final_reward = 0

        if not env.is_multi_player:
            while not done:
                a = monte_carlo_random_rollout(env, num_rollouts)
                _, reward, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                final_reward = reward
        else:
            while not done:
                if env.current_player == env.agent_player:
                    a = monte_carlo_random_rollout(env, num_rollouts)
                else:
                    mask = env.get_action_mask()
                    a = env.action_space.sample(mask=mask)

                _, reward, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                final_reward = reward

        rewards.append(final_reward)

        if (episode + 1) % 100 == 0:
            last_100 = np.array(rewards[-100:])
            wins   = np.sum(last_100 ==  1)
            draws  = np.sum(last_100 ==  0)
            losses = np.sum(last_100 == -1)

            wins_rate.append(wins)
            draws_rate.append(draws)
            losses_rate.append(losses)

            print(f"[{episode+1}/{num_episodes}] Win: {wins}% | Draw: {draws}% | Loss: {losses}%")

    # --- Plot ---
    x = np.arange(100, num_episodes + 1, 100)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, wins_rate,   label="Victoires %", color="green")
    ax.plot(x, draws_rate,  label="Nuls %",      color="orange")
    ax.plot(x, losses_rate, label="Défaites %",  color="red")
    ax.set_xlabel("Épisode")
    ax.set_ylabel("% sur 100 épisodes glissants")
    ax.set_title(f"Monte Carlo Rollout — {env.unwrapped} | rollouts={num_rollouts}")
    ax.legend()
    plt.tight_layout()

    save_dir = f"{settings.training_logs_path}/MonteCarloRandomRollout/{env.unwrapped}"
    os.makedirs(save_dir, exist_ok=True)
    path = f"{save_dir}/monte_carlo_{env.unwrapped}.png"
    plt.savefig(path)
    plt.close()
    print(f"Plot saved → {path}")

if __name__ == "__main__":
    env = LineWorldEnv()  # ou TicTacToeEnv(), GridWorldEnv()...
    run_monte_carlo(env, num_episodes=1_000, num_rollouts=50)

    env = GridWorldEnv()  # ou TicTacToeEnv(), GridWorldEnv()...
    run_monte_carlo(env, num_episodes=1_000, num_rollouts=50)


    env = TicTacToeEnv()  # ou TicTacToeEnv(), GridWorldEnv()...
    run_monte_carlo(env, num_episodes=1_000, num_rollouts=50)

    env = QuartoEnv()  # ou TicTacToeEnv(), GridWorldEnv()...
    run_monte_carlo(env, num_episodes=1_000, num_rollouts=50)