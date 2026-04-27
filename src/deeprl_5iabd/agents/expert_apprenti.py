import os
import numpy as np
from tqdm import tqdm
from deeprl_5iabd.envs.tictactoe import TicTacToeEnv
from deeprl_5iabd.envs.line_world import LineWorldEnv
from deeprl_5iabd.envs.grid_world import GridWorldEnv
from deeprl_5iabd.envs.quarto import QuartoEnv
from deeprl_5iabd.config import settings
from gymnasium import Env
import torch
import torch.nn as nn
import torch.optim as optim


def save_dataset(states, q_values, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, states=states, q_values=q_values)
    return path

def load_dataset(path):
    data = np.load(path)
    return data["states"], data["q_values"]

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved at: {path}")


def load_model(model_class, env, path, device="cpu"):
    model = model_class(env)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from: {path}")
    return model


def collect_mcrr_dataset(env, num_episodes=10_000, num_simulations=50):

    print(f"Collecting dataset for {env.unwrapped}")

    states = []
    q_targets = []

    for _ in tqdm(range(num_episodes)):
        obs, _ = env.reset()

        terminated = False
        truncated = False

        while not (terminated or truncated):
            q_values = mcrr_q_values(env, num_simulations)
            best_action_idx = np.argmax(q_values)
            states.append(obs.copy())
            q_targets.append(q_values)

            obs, reward, terminated, truncated, _ = env.step(best_action_idx)

    return np.array(states, dtype=np.float32), np.array(q_targets, dtype=np.float32)


def mcrr_q_values(env: Env, num_simulations: int):

    mask = env.get_action_mask()
    action_mean_rewards = np.full(len(mask), -np.inf)

    valid_actions = np.where(mask == 1)[0]
    action_mean_rewards[valid_actions] = 0

    if len(valid_actions) == 0:
        return

    a_resource = num_simulations // len(valid_actions)

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

    invalid_actions = np.where(mask == 0)[0]
    action_mean_rewards[invalid_actions] = np.min(action_mean_rewards[valid_actions])

    return action_mean_rewards


class QNet(nn.Module):
    def __init__(self, env):
        super().__init__()

        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def train_expert_apprenti(x_data, y_data, env, epochs=1_000, batch_size=2048, lr=1e-3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = QNet(env).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    x_data = torch.tensor(x_data, dtype=torch.float32).to(device)
    y_data = torch.tensor(y_data, dtype=torch.float32).to(device)

    n = len(x_data)

    for epoch in tqdm(range(epochs), desc="Training epochs"):

        total_loss = 0.0

        indices = torch.randperm(n, device=device)

        for i in range(0, n, batch_size):

            batch_idx = indices[i:i + batch_size]

            x_batch = x_data[batch_idx]
            y_batch = y_data[batch_idx]

            q_pred = model(x_batch)

            loss = loss_fn(q_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, (n // batch_size))
        tqdm.write(f"Epoch {epoch} | Loss = {avg_loss:.6f}")

    return model

def demo_model(model, env):
    obs, _ = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        if (env.current_player == env.agent_player):
            q_values = model(torch.tensor(obs, dtype=torch.float32))
            best_action_idx = torch.argmax(q_values).item()
            obs, reward, terminated, truncated, _ = env.step(best_action_idx)
        else:
            mask = env.get_action_mask()
            a = env.action_space.sample(mask=mask)
            obs, reward, terminated, truncated, _ = env.step(a)
    
    return reward

if __name__ == "__main__":

    NUM_EPISODES = 100_000
    NUM_SIMULATIONS = 100

    env = QuartoEnv()
    # dataset_path = f"{settings.training_logs_dir}/expert_apprenti/{env.unwrapped}/dataset_{NUM_EPISODES}_sim{NUM_SIMULATIONS}.npz"
    # model_path = f"{settings.models_dir}/expert_apprenti/{env.unwrapped}/model_{NUM_EPISODES}_sim{NUM_SIMULATIONS}.pt"

    # print("Generating dataset with MCRR...")
    # states, q_values = collect_mcrr_dataset(
    #     env,
    #     num_episodes=NUM_EPISODES,
    #     num_simulations=NUM_SIMULATIONS
    # )

    # save_dataset(states, q_values, dataset_path)
    # print(f"Dataset saved at: {dataset_path}")

    # print("Training model...")
    # states, q_values = load_dataset(dataset_path)

    # model = train_expert_apprenti(states, q_values, env)
    # save_model(model, model_path)
    # print("Model trained and saved in", model_path)

    # print("Loading model...")
    # model = load_model(QNet, env, model_path)
    # total_reward = 0

