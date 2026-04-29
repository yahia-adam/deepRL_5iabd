import os
import json
import time
import numpy as np
from tqdm import tqdm
from gymnasium import Env
from gymnasium.wrappers import RecordVideo
from torch.distributions import Categorical

import torch
import torch.nn as nn
import torch.optim as optim

from deeprl_5iabd.envs.line_world import LineWorldEnv
from deeprl_5iabd.envs.grid_world import GridWorldEnv
from deeprl_5iabd.envs.tictactoe import TicTacToeEnv
from deeprl_5iabd.envs.quarto import QuartoEnv
from deeprl_5iabd.config import settings
from deeprl_5iabd.helper import plot_metric


SEEDS = (42, 123, 7)


def save_dataset(states, q_values, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, states=states, q_values=q_values)
    return path


def load_dataset(path):
    data = np.load(path)
    return data["states"], data["q_values"]


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
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved at: {path}")


def load_model(env, path, device="cpu"):
    model = QNet(env)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from: {path}")
    return model

def mcrr_q_values(env: Env, num_simulations: int):
    mask = env.get_action_mask()
    action_mean_rewards = np.full(len(mask), -np.inf)

    valid_actions = np.where(mask == 1)[0]
    action_mean_rewards[valid_actions] = 0

    if len(valid_actions) == 0:
        return action_mean_rewards

    a_resource = num_simulations // len(valid_actions)

    for test_action in valid_actions:
        for _ in range(a_resource):
            new_env = env.determinize()
            total_reward = 0

            _, reward, terminated, truncated, _ = new_env.step(test_action)
            total_reward += reward

            while not (terminated or truncated):
                m = new_env.get_action_mask()
                a = new_env.action_space.sample(mask=m)
                _, reward, terminated, truncated, _ = new_env.step(a)
                total_reward += reward

            action_mean_rewards[test_action] += total_reward

        action_mean_rewards[test_action] /= a_resource

    invalid_actions = np.where(mask == 0)[0]
    action_mean_rewards[invalid_actions] = np.min(action_mean_rewards[valid_actions])

    return action_mean_rewards


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
            best_action_idx = int(np.argmax(q_values))
            states.append(obs.copy())
            q_targets.append(q_values)
            obs, reward, terminated, truncated, _ = env.step(best_action_idx)

    return np.array(states, dtype=np.float32), np.array(q_targets, dtype=np.float32)


def train_expert_apprenti(
    x_data,
    y_data,
    env,
    epochs: int = 1_000,
    batch_size: int = 2048,
    lr: float = 1e-3,
    num_episodes_dataset: int = 10_000,
    num_simulations: int = 50,
    checkpoints: tuple = (100, 500, 1_000),
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = QNet(env).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    x_tensor = torch.tensor(x_data, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_data, dtype=torch.float32).to(device)

    n = len(x_tensor)

    agent_name = f"expert_apprenti_ep{num_episodes_dataset}_sim{num_simulations}"

    loss_history = np.zeros(epochs + 1)
    time_per_epoch_history = np.zeros(epochs + 1)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        total_loss = 0.0
        indices = torch.randperm(n, device=device)

        for i in range(0, n, batch_size):
            batch_idx = indices[i:i + batch_size]
            x_batch = x_tensor[batch_idx]
            y_batch = y_tensor[batch_idx]

            q_pred = model(x_batch)
            loss = loss_fn(q_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, n // batch_size)
        epoch_time = time.perf_counter() - t0

        loss_history[epoch] = avg_loss
        time_per_epoch_history[epoch] = epoch_time

        if epoch % 10 == 0:
            print(
                f"[{agent_name} | {env.unwrapped} | seed={seed}] "
                f"Epoch {epoch}/{epochs} | Loss={avg_loss:.6f} | "
                f"Time={epoch_time * 1000:.1f}ms"
            )

        if epoch in checkpoints:
            model_dir = f"{settings.models_path}/expert_apprenti/{env.unwrapped}/seed_{seed}"
            os.makedirs(model_dir, exist_ok=True)
            model_path = f"{model_dir}/model_{agent_name}_epoch{epoch}.pt"
            save_model(model, model_path)

    plot_dir = f"{settings.training_logs_dir}/expert_apprenti/{env.unwrapped}/seed_{seed}/train"
    os.makedirs(plot_dir, exist_ok=True)

    plot_metric(values=loss_history, save_dir=plot_dir, window_size=10,
                exp_name=f"training_loss_{agent_name}_env_{env.unwrapped}", metric_name="loss")
    plot_metric(values=time_per_epoch_history, save_dir=plot_dir, window_size=10,
                exp_name=f"time_per_epoch_{agent_name}_env_{env.unwrapped}", metric_name="time_per_epoch")

    train_results = {
        "env": str(env.unwrapped),
        "agent": agent_name,
        "seed": seed,
        "epochs": epochs,
        "hyperparameters": {
            "lr": lr,
            "batch_size": batch_size,
            "num_episodes_dataset": num_episodes_dataset,
            "num_simulations": num_simulations,
        },
        "summary": {
            "final_loss": float(loss_history[epochs]),
            "min_loss": float(np.min(loss_history[1:])),
            "mean_loss": float(np.mean(loss_history[1:])),
            "mean_time_per_epoch_ms": float(np.mean(time_per_epoch_history[1:]) * 1000),
        },
        "epochs": [
            {
                "epoch": int(ep),
                "loss": float(loss_history[ep]),
                "time_per_epoch_ms": float(time_per_epoch_history[ep] * 1000),
            }
            for ep in range(1, epochs + 1)
        ],
    }

    json_path = f"{plot_dir}/{agent_name}_epoch{epochs}.json"
    with open(json_path, "w") as f:
        json.dump(train_results, f, indent=2)
    print(f"Training metrics saved: {json_path}")

    return model

def opponent_step(env):
    mask = env.get_action_mask()
    action = env.action_space.sample(mask=mask)
    return env.step(action)


def eval_agent(
    env: Env,
    num_episodes: int = 1_000,
    model_name: str = "model_expert_apprenti_ep10000_sim50_epoch1000.pt",
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(env, f"{settings.models_path}/expert_apprenti/{env.unwrapped}/seed_{seed}/{model_name}", device)

    is_multi = getattr(env, "is_multi_player", False)

    rewards_history = np.zeros(num_episodes)
    n_steps_history = np.zeros(num_episodes, dtype=int)
    time_per_move_history = np.zeros(num_episodes)

    base = model_name.replace("model_", "").replace(".pt", "")
    agent_name, epoch_str = base.rsplit("_epoch", 1)
    checkpoint_epoch = int(epoch_str)

    print(f"[{agent_name} | {env.unwrapped} | seed={seed}] Starting eval "
          f"num_episodes={num_episodes} checkpoint_epoch={checkpoint_epoch}")

    for i in range(num_episodes):
        n_step = 0
        episode_time = 0.0

        with torch.no_grad():
            obs, _ = env.reset()
            done = False
            truncated = False
            reward = 0.0

            while not (done or truncated):
                if is_multi and env.current_player != env.agent_player:
                    mask = env.get_action_mask()
                    action = env.action_space.sample(mask=mask)
                    obs, reward, done, truncated, _ = env.step(action)
                    n_step += 1
                else:
                    t0 = time.perf_counter()
                    q_values = model(torch.tensor(obs, dtype=torch.float32).to(device))
                    action = int(torch.argmax(q_values).item())
                    episode_time += time.perf_counter() - t0

                    obs, reward, done, truncated, _ = env.step(action)
                    n_step += 1

        rewards_history[i] = reward
        n_steps_history[i] = n_step
        time_per_move_history[i] = episode_time / max(n_step, 1)

        if (i + 1) % 100 == 0:
            recent_rewards = rewards_history[max(0, i - 99):i + 1]
            win_rate = np.mean(recent_rewards == 1) * 100
            loss_rate = np.mean(recent_rewards == -1) * 100
            print(
                f"[{agent_name} | {env.unwrapped} | seed={seed}] Episode {i + 1} | "
                f"Win={win_rate:.0f}% Lose={loss_rate:.0f}% | "
                f"Time/move={time_per_move_history[i] * 1000:.2f}ms"
            )

    plot_dir = f"{settings.training_logs_dir}/expert_apprenti/{env.unwrapped}/seed_{seed}/eval"
    os.makedirs(plot_dir, exist_ok=True)

    plot_metric(values=rewards_history, save_dir=plot_dir, window_size=0,
                exp_name=f"{agent_name}_epoch{checkpoint_epoch}_env_{env.unwrapped}", metric_name="winrate")
    plot_metric(values=n_steps_history, save_dir=plot_dir, window_size=0,
                exp_name=f"{agent_name}_epoch{checkpoint_epoch}_env_{env.unwrapped}", metric_name="nbr_steps")
    plot_metric(values=time_per_move_history, save_dir=plot_dir, window_size=0,
                exp_name=f"{agent_name}_epoch{checkpoint_epoch}_env_{env.unwrapped}", metric_name="time_per_move")

    results = {
        "env": str(env.unwrapped),
        "agent": agent_name,
        "checkpoint_epoch": checkpoint_epoch,
        "seed": seed,
        "num_episodes": num_episodes,
        "summary": {
            "mean_reward": float(np.mean(rewards_history)),
            "win_rate": float(np.mean(rewards_history == 1)),
            "loss_rate": float(np.mean(rewards_history == -1)),
            "draw_rate": float(np.mean(rewards_history == 0)),
            "mean_steps": float(np.mean(n_steps_history)),
            "std_steps": float(np.std(n_steps_history)),
            "mean_time_per_move_ms": float(np.mean(time_per_move_history) * 1000),
        },
        "episodes": [
            {
                "episode": int(i),
                "reward": float(rewards_history[i]),
                "n_steps": int(n_steps_history[i]),
                "time_per_move_ms": float(time_per_move_history[i] * 1000),
            }
            for i in range(num_episodes)
        ],
    }

    json_path = f"{plot_dir}/{agent_name}_epoch{checkpoint_epoch}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    s = results["summary"]
    print(
        f"[EVAL {env.unwrapped}/{agent_name}@epoch{checkpoint_epoch} | seed={seed}] "
        f"win={s['win_rate']:.2%} loss={s['loss_rate']:.2%} draw={s['draw_rate']:.2%} "
        f"mean_steps={s['mean_steps']:.2f} -> {json_path}"
    )
    return results


def eval_all_models_for_env(env, seed, num_episodes_dataset=10_000, num_simulations=50):
    agent_tag = f"expert_apprenti_ep{num_episodes_dataset}_sim{num_simulations}"
    for epoch in (100, 500, 1_000):
        eval_agent(
            env,
            num_episodes=1_000,
            model_name=f"model_{agent_tag}_epoch{epoch}.pt",
            seed=seed,
        )


def wrap_video(env, seed, episode_num_trigger):
    video_env = RecordVideo(
        env,
        video_folder=f"{settings.videos_dir}/expert_apprenti/{env.unwrapped}/seed_{seed}/eval/",
        episode_trigger=lambda ep: ep % episode_num_trigger == 0,
    )
    video_env.state_id = env.state_id
    video_env.get_action_mask = env.get_action_mask
    if hasattr(env, "determinize"):
        video_env.determinize = env.determinize
    if hasattr(env, "agent_player"):
        video_env.agent_player = env.agent_player
    if hasattr(env, "is_multi_player"):
        video_env.is_multi_player = env.is_multi_player
    type(video_env).current_player = property(
        lambda self: env.current_player,
        lambda self, v: setattr(env, "current_player", v)
    )
    return video_env


if __name__ == "__main__":
    NUM_EPISODES_DATASET = 10_000
    NUM_SIMULATIONS = 100
    EPOCHS = 1_000

    env_classes = [LineWorldEnv, GridWorldEnv, TicTacToeEnv, QuartoEnv]

    for seed in SEEDS:
        for EnvCls in env_classes:
            env = EnvCls()

            #  Dataset 
            dataset_path = (
                f"{settings.training_logs_dir}/expert_apprenti/{env.unwrapped}/seed_{seed}/"
                f"dataset_ep{NUM_EPISODES_DATASET}_sim{NUM_SIMULATIONS}.npz"
            )
            if not os.path.exists(dataset_path):
                states, q_values = collect_mcrr_dataset(
                    env, num_episodes=NUM_EPISODES_DATASET, num_simulations=NUM_SIMULATIONS
                )
                save_dataset(states, q_values, dataset_path)
                print(f"Dataset saved at: {dataset_path}")
            else:
                print(f"Dataset already exists, loading: {dataset_path}")

            states, q_values = load_dataset(dataset_path)
            env.close()

            #  Train 
            env_train = EnvCls()
            print(f"\n{'=' * 60}\nTRAIN {env_train.unwrapped} | seed={seed}\n{'=' * 60}")
            train_expert_apprenti(
                states, q_values, env_train,
                epochs=EPOCHS,
                num_episodes_dataset=NUM_EPISODES_DATASET,
                num_simulations=NUM_SIMULATIONS,
                checkpoints=(100, 500, 1_000),
                seed=seed,
            )
            env_train.close()

            #  Eval 
            env_eval = EnvCls(render_mode="rgb_array")
            video_env_eval = wrap_video(env_eval, seed, episode_num_trigger=100)
            print(f"\n{'=' * 60}\nEVAL {env_eval.unwrapped} | seed={seed}\n{'=' * 60}")
            eval_all_models_for_env(
                video_env_eval, seed,
                num_episodes_dataset=NUM_EPISODES_DATASET,
                num_simulations=NUM_SIMULATIONS,
            )
            video_env_eval.close()