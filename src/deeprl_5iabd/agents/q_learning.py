import os
import json
import time
import random
import pickle
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from deeprl_5iabd.helper import plot_metric
from deeprl_5iabd.envs.line_world import LineWorldEnv
from deeprl_5iabd.envs.grid_world import GridWorldEnv
from deeprl_5iabd.envs.tictactoe import TicTacToeEnv
from deeprl_5iabd.config import settings


SEEDS = (42, 123, 7)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def opponent_step(env):
    mask = env.get_action_mask()
    action = env.action_space.sample(mask=mask)
    return env.step(action)


def q_learning(
    env: gym.Env,
    num_episodes: int = 100_000,
    lr: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.00005,
    checkpoints: tuple = (1_000, 10_000, 100_000),
    seed: int = 42,
):
    set_seed(seed)

    rng = np.random.default_rng(seed)

    is_multi = getattr(env, "is_multi_player", False)

    if isinstance(env.unwrapped, TicTacToeEnv):
        num_states = 3 ** 9
    else:
        num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))

    agent_name = "q_learning"

    rewards_history = np.zeros(num_episodes + 1)
    loss_history = np.zeros(num_episodes + 1)
    nbr_steps_history = np.zeros(num_episodes + 1)
    time_per_move_history = np.zeros(num_episodes + 1)

    current_lr = lr
    epsilon_start = epsilon

    for episode in range(1, num_episodes + 1):

        env.reset()
        state = env.state_id()
        done = False
        truncated = False
        n_step = 0
        final_reward = 0.0
        episode_td_errors = []

        episode_start = time.perf_counter()

        while not (done or truncated):

            if is_multi:
                while not (done or truncated) and env.current_player != env.agent_player:
                    _, final_reward, done, truncated, _ = opponent_step(env)
                    n_step += 1

            while not (done or truncated) and (not is_multi or env.current_player == env.agent_player):
                mask = env.get_action_mask()
                valid_actions = np.where(mask == 1)[0]

                if rng.random() < epsilon:
                    action = int(rng.choice(valid_actions))
                else:
                    q_masked = np.full(num_actions, -np.inf)
                    q_masked[valid_actions] = Q[state, valid_actions]
                    action = int(np.argmax(q_masked))

                _, final_reward, done, truncated, _ = env.step(action)
                new_state = env.state_id()
                n_step += 1

                td_target = final_reward + gamma * np.max(Q[new_state, :]) * (not done)
                td_error = td_target - Q[state, action]
                Q[state, action] += current_lr * td_error
                episode_td_errors.append(abs(td_error))

                state = new_state

        episode_time = time.perf_counter() - episode_start
        time_per_move_history[episode] = episode_time / max(n_step, 1)

        rewards_history[episode] = final_reward
        nbr_steps_history[episode] = n_step
        loss_history[episode] = float(np.mean(episode_td_errors)) if episode_td_errors else 0.0

        epsilon = max(epsilon - epsilon_decay, 0.0)
        if epsilon == 0.0:
            current_lr = 0.0001

        if episode % 100 == 0:
            recent_rewards = rewards_history[max(1, episode - 100):episode + 1]
            recent_tpm = time_per_move_history[max(1, episode - 100):episode + 1]
            win_rate = np.mean(recent_rewards == 1) * 100
            loss_rate = np.mean(recent_rewards == -1) * 100
            print(
                f"[{agent_name} | {env.unwrapped} | seed={seed}] Episode {episode} | "
                f"Win={win_rate:.0f}% Lose={loss_rate:.0f}% | "
                f"TD-Error={loss_history[episode]:.4f} | "
                f"Epsilon={epsilon:.4f} | "
                f"Time/move={np.mean(recent_tpm) * 1000:.2f}ms"
            )

        if episode in checkpoints:
            model_dir = f"{settings.models_path}/q_learning/{env.unwrapped}/seed_{seed}"
            os.makedirs(model_dir, exist_ok=True)
            model_path = f"{model_dir}/policy_{agent_name}_{episode}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(Q, f)
            print(f"Model saved: {model_path}")

    plot_dir = f"{settings.training_logs_dir}/q_learning/{env.unwrapped}/seed_{seed}/train"
    os.makedirs(plot_dir, exist_ok=True)

    plot_metric(values=rewards_history, save_dir=plot_dir, window_size=100,
                exp_name=f"{agent_name}_env_{env.unwrapped}", metric_name="winrate")
    plot_metric(values=loss_history, save_dir=plot_dir, window_size=100,
                exp_name=f"training_loss_{agent_name}_env_{env.unwrapped}", metric_name="loss")
    plot_metric(values=nbr_steps_history, save_dir=plot_dir, window_size=100,
                exp_name=f"nbr_steps_{agent_name}_env_{env.unwrapped}", metric_name="nbr_steps")
    plot_metric(values=time_per_move_history, save_dir=plot_dir, window_size=100,
                exp_name=f"time_per_move_{agent_name}_env_{env.unwrapped}", metric_name="time_per_move")

    train_results = {
        "env": str(env.unwrapped),
        "agent": agent_name,
        "seed": seed,
        "num_episodes": num_episodes,
        "hyperparameters": {
            "lr": lr,
            "gamma": gamma,
            "epsilon_start": epsilon_start,
            "epsilon_decay": epsilon_decay,
        },
        "summary": {
            "mean_reward": float(np.mean(rewards_history[1:])),
            "win_rate": float(np.mean(rewards_history[1:] == 1)),
            "loss_rate": float(np.mean(rewards_history[1:] == -1)),
            "draw_rate": float(np.mean(rewards_history[1:] == 0)),
            "mean_td_error": float(np.mean(loss_history[1:])),
            "mean_steps": float(np.mean(nbr_steps_history[1:])),
            "std_steps": float(np.std(nbr_steps_history[1:])),
            "mean_time_per_move_ms": float(np.mean(time_per_move_history[1:]) * 1000),
        },
        "episodes": [
            {
                "episode": int(ep),
                "reward": float(rewards_history[ep]),
                "td_error": float(loss_history[ep]),
                "n_steps": int(nbr_steps_history[ep]),
                "time_per_move_ms": float(time_per_move_history[ep] * 1000),
            }
            for ep in range(1, num_episodes + 1)
        ],
    }

    json_path = f"{plot_dir}/{agent_name}_{num_episodes}.json"
    with open(json_path, "w") as f:
        json.dump(train_results, f, indent=2)
    print(f"Training metrics saved: {json_path}")

    return Q


def eval_agent(env, num_episodes=1_000, model_name="policy_q_learning_1000.pkl", seed: int = 42):
    with open(f"{settings.models_path}/q_learning/{env.unwrapped}/seed_{seed}/{model_name}", "rb") as f:
        Q = pickle.load(f)

    num_actions = env.action_space.n
    is_multi = getattr(env, "is_multi_player", False)

    rewards_history = np.zeros(num_episodes)
    n_steps_history = np.zeros(num_episodes, dtype=int)
    time_per_move_history = np.zeros(num_episodes)

    for i in range(num_episodes):
        n_step = 0
        env.reset()
        state = env.state_id()
        done = False
        truncated = False
        reward = 0.0

        episode_start = time.perf_counter()

        while not (done or truncated):

            if is_multi:
                while not (done or truncated) and env.current_player != env.agent_player:
                    _, reward, done, truncated, _ = opponent_step(env)
                    n_step += 1

            while not (done or truncated) and (not is_multi or env.current_player == env.agent_player):
                mask = env.get_action_mask()
                valid_actions = np.where(mask == 1)[0]

                q_masked = np.full(num_actions, -np.inf)
                q_masked[valid_actions] = Q[state, valid_actions]
                action = int(np.argmax(q_masked))

                _, reward, done, truncated, _ = env.step(action)
                state = env.state_id()
                n_step += 1

        episode_time = time.perf_counter() - episode_start
        rewards_history[i] = reward
        n_steps_history[i] = n_step
        time_per_move_history[i] = episode_time / max(n_step, 1)

    base = model_name.replace("policy_", "").replace(".pkl", "")
    agent_name, checkpoint_str = base.rsplit("_", 1)
    checkpoint = int(checkpoint_str)

    plot_dir = f"{settings.training_logs_dir}/q_learning/{env.unwrapped}/seed_{seed}/eval"
    os.makedirs(plot_dir, exist_ok=True)

    plot_metric(values=rewards_history, save_dir=plot_dir, window_size=0,
                exp_name=f"{agent_name}_{checkpoint}_env_{env.unwrapped}", metric_name="winrate")
    plot_metric(values=n_steps_history, save_dir=plot_dir, window_size=0,
                exp_name=f"{agent_name}_{checkpoint}_env_{env.unwrapped}", metric_name="nbr_steps")
    plot_metric(values=time_per_move_history, save_dir=plot_dir, window_size=0,
                exp_name=f"{agent_name}_{checkpoint}_env_{env.unwrapped}", metric_name="time_per_move")

    results = {
        "env": str(env.unwrapped),
        "agent": agent_name,
        "checkpoint": checkpoint,
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

    json_path = f"{plot_dir}/{agent_name}_{checkpoint}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    s = results["summary"]
    print(
        f"[EVAL {env.unwrapped}/{agent_name}@{checkpoint} | seed={seed}] "
        f"win={s['win_rate']:.2%} loss={s['loss_rate']:.2%} draw={s['draw_rate']:.2%} "
        f"mean_steps={s['mean_steps']:.2f} -> {json_path}"
    )
    return results


def train_all_for_env(env, seed):
    q_learning(env, num_episodes=100_000, seed=seed)


def eval_all_models_for_env(env, seed):
    for n in (1_000, 10_000, 100_000):
        eval_agent(env, num_episodes=1_000, model_name=f"policy_q_learning_{n}.pkl", seed=seed)


def wrap_video(env, mode, seed, episode_num_trigger):
    video_env = RecordVideo(
        env,
        video_folder=f"{settings.videos_dir}/q_learning/{env.unwrapped}/seed_{seed}/{mode}/",
        episode_trigger=lambda ep: ep % episode_num_trigger == 0,
    )
    video_env.state_id = env.state_id
    video_env.get_action_mask = env.get_action_mask
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
    env_classes = [LineWorldEnv, GridWorldEnv, TicTacToeEnv]

    for seed in SEEDS:
        for EnvCls in env_classes:
            #  TRAIN 
            env_train = EnvCls(render_mode="rgb_array")
            video_env_train = wrap_video(env_train, "train", seed, 10_000)
            print(f"\n{'=' * 60}\nTRAIN {env_train.unwrapped} | seed={seed}\n{'=' * 60}")
            train_all_for_env(video_env_train, seed)
            video_env_train.close()

            #  EVAL 
            env_eval = EnvCls(render_mode="rgb_array")
            video_env_eval = wrap_video(env_eval, "eval", seed, 100)
            print(f"\n{'=' * 60}\nEVAL {env_eval.unwrapped} | seed={seed}\n{'=' * 60}")
            eval_all_models_for_env(video_env_eval, seed)
            video_env_eval.close()