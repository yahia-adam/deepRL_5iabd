import os
import numpy as np
from gymnasium import Env
import matplotlib.pyplot as plt
import pickle
from deeprl_5iabd.config import settings

def q_learning(
    env: Env,
    learning_rate: float = 0.001,
    gamma: float = 0.9,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.0001,
    num_episodes: int = 100_000
):
    Q = np.zeros((env.observation_space.shape[0], env.action_space.n))
    rng = np.random.default_rng()
    reward_per_episode = np.zeros(num_episodes)

    for i in range(num_episodes):
        _, _ = env.reset()
        state = env.state_id()
        terminated = False
        truncated = False

        while not terminated and not truncated:
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            _, reward, terminated, truncated, _ = env.step(action)
            new_state = env.state_id()

            Q[state, action] = Q[state, action] + learning_rate * (
                reward + gamma * np.max(Q[new_state, :]) - Q[state, action]
            )
            state = new_state
            reward_per_episode[i] += reward

        epsilon = max(epsilon - epsilon_decay, 0.0)
        if epsilon == 0.0:
            learning_rate = 0.0001

    sum_rewards = np.zeros(num_episodes)
    for t in range(num_episodes):
        sum_rewards[t] = np.sum(reward_per_episode[max(0, t - 100):t + 1])

    plt.plot(sum_rewards)
    save_dir = f"{settings.training_logs_path}/q_learning/{env.unwrapped}"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/plot_reward.png")

    model_dir = f"{settings.models_path}/q_learning/{env.unwrapped}"
    os.makedirs(model_dir, exist_ok=True)
    with open(f"{model_dir}/model.pkl", "wb") as f:
        pickle.dump(Q, f)

    env.close()

def q_learning_tictactoe(
    env: Env,
    learning_rate: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.00005,
    num_episodes: int = 300_000
):
    NUM_STATES = 3 ** 9
    Q = np.zeros((NUM_STATES, env.action_space.n))
    rng = np.random.default_rng()
    reward_per_episode = np.zeros(num_episodes)

    for i in range(num_episodes):
        env.reset()
        agent_player = env.agent_player
        state = env.state_id()
        terminated = False
        truncated = False

        while not terminated and not truncated:
            mask = env.get_action_mask()
            valid_actions = np.where(mask == 1)[0]

            if env.current_player == agent_player:
                # Coup de l'agent Q (epsilon-greedy)
                if rng.random() < epsilon:
                    action = rng.choice(valid_actions)
                else:
                    q_masked = np.full(env.action_space.n, -np.inf)
                    q_masked[valid_actions] = Q[state, valid_actions]
                    action = int(np.argmax(q_masked))

                _, reward, terminated, truncated, _ = env.step(action)
                new_state = env.state_id()

                Q[state, action] = Q[state, action] + learning_rate * (
                    reward + gamma * np.max(Q[new_state, :]) - Q[state, action]
                )
                state = new_state
                reward_per_episode[i] += reward

            else:
                # Coup de l'adversaire random
                action = rng.choice(valid_actions)
                _, _, terminated, truncated, _ = env.step(action)

        epsilon = max(epsilon - epsilon_decay, 0.0)
        if epsilon == 0.0:
            learning_rate = 0.0001

    sum_rewards = np.zeros(num_episodes)
    for t in range(num_episodes):
        sum_rewards[t] = np.sum(reward_per_episode[max(0, t - 100):t + 1])

    plt.plot(sum_rewards)
    plt.savefig(f"q_learning_{env}.png")

    with open(f"q_learning_{env}.pkl", "wb") as f:
        pickle.dump(Q, f)
    env.close()