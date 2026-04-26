import os
import numpy as np
from gymnasium import Env
import matplotlib.pyplot as plt
import pickle
from torch.utils.tensorboard import SummaryWriter
from deeprl_5iabd.config import settings
from deeprl_5iabd.envs.line_world import LineWorldEnv
from deeprl_5iabd.envs.tictactoe import TicTacToeEnv


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

    tb_log_dir = f"{settings.training_logs_path}/q_learning/{env.unwrapped}/tensorboard"
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(tb_log_dir)

    save_checkpoints = [1_000, 10_000, 100_000, 1_000_000]

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

        writer.add_scalar("Reward/Episode", reward_per_episode[i], i)
        writer.add_scalar(
            "Reward/Mean_100",
            np.mean(reward_per_episode[max(0, i - 100):i + 1]),
            i
        )
        writer.add_scalar("Hyperparameters/Epsilon", epsilon, i)
        writer.add_scalar("Hyperparameters/Learning_rate", learning_rate, i)

        if (i + 1) in save_checkpoints:
            model_dir = f"{settings.models_path}/q_learning/{env.unwrapped}"
            os.makedirs(model_dir, exist_ok=True)

            with open(f"{model_dir}/model_{i+1}.pkl", "wb") as f:
                pickle.dump(Q, f)

    sum_rewards = np.zeros(num_episodes)
    for t in range(num_episodes):
        sum_rewards[t] = np.sum(reward_per_episode[max(0, t - 100):t + 1])

    plt.plot(sum_rewards)
    save_dir = f"{settings.training_logs_path}/q_learning/{env.unwrapped}"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/plot_reward.png")

    model_dir = f"{settings.models_path}/q_learning/{env.unwrapped}"
    os.makedirs(model_dir, exist_ok=True)
    with open(f"{model_dir}/model_final.pkl", "wb") as f:
        pickle.dump(Q, f)

    writer.close()
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
    save_dir = f"{settings.training_logs_path}/q_learning_tictactoe/"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/plot_reward.png")

    model_dir = f"{settings.models_path}/q_learning_tictactoe/"
    os.makedirs(model_dir, exist_ok=True)
    with open(f"{model_dir}/model.pkl", "wb") as f:
        pickle.dump(Q, f)

    env.close()



# if __name__ == "__main__":
#     # env = LineWorldEnv()
#     # q_learning(env, num_episodes=1_000_000)

#     env = TicTacToeEnv()
#     q_learning_tictactoe(env, num_episodes=1_000_000)

def play_and_plot(model_path, expname, num_episodes=100):
    """
    Charge une Q-table et joue num_episodes parties.
    Stocke les rewards et génère un plot matplotlib.
    """

    with open(model_path, "rb") as f:
        Q = pickle.load(f)

    rng = np.random.default_rng()
    rewards = np.zeros(num_episodes)

    for i in range(num_episodes):
        _, _ = env.reset()
        state = env.state_id()
        terminated = False
        truncated = False

        while not terminated and not truncated:

            action = int(np.argmax(Q[state, :]))

            _, reward, terminated, truncated, _ = env.step(action)

            state = env.state_id()

            rewards[i] += reward

    env.close()

    plt.figure()
    plt.plot(rewards, marker="o")
    plt.title(f"Evaluation over {num_episodes} episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.show()

    save_dir = f"{settings.training_logs_path}/{expname}/eval"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/plot_reward.png")

    print(f"Average reward: {np.mean(rewards):.3f}")

    return rewards

if __name__ == "__main__":
    env = LineWorldEnv()

    play_and_plot(
        expname="q_learning_linewoard_e0.01_lr0.001",
        model_path="/home/adam/Documents/esgi/drl/deepRL_5iabd/experimentation_logs/models/q_learning/LineWorldEnv/model_1000.pkl",
        env=env,
        num_episodes=100,
    )