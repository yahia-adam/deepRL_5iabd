import os
import json
import time
import numpy as np
import gymnasium as gym
from gymnasium import Env
from gymnasium.wrappers import RecordVideo

from deeprl_5iabd.helper import plot_metric
from deeprl_5iabd.envs.line_world import LineWorldEnv
from deeprl_5iabd.envs.grid_world import GridWorldEnv
from deeprl_5iabd.envs.tictactoe import TicTacToeEnv
from deeprl_5iabd.envs.quarto import QuartoEnv
from deeprl_5iabd.config import settings


SEEDS = (42, 123, 7)


def monte_carlo_random_rollout(env: Env, num_simulations: int):
    mask = env.get_action_mask()
    action_mean_rewards = np.full(len(mask), -np.inf)

    valid_actions = np.where(mask == 1)[0]
    action_mean_rewards[valid_actions] = 0

    if len(valid_actions) == 0:
        return None

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

    return int(np.argmax(action_mean_rewards))


def eval_agent(
    env: Env,
    num_episodes: int = 1_000,
    num_simulations: int = 100,
    seed: int = 42,
):
    np.random.seed(seed)

    agent_name = f"mcrr_sim{num_simulations}"
    is_multi = getattr(env, "is_multi_player", False)

    rewards_history = np.zeros(num_episodes)
    n_steps_history = np.zeros(num_episodes, dtype=int)
    time_per_move_history = np.zeros(num_episodes)

    print(f"[{agent_name} | {env.unwrapped} | seed={seed}] Starting eval "
          f"num_episodes={num_episodes} num_simulations={num_simulations}")

    for i in range(num_episodes):
        env.reset()
        done = False
        truncated = False
        reward = 0.0
        n_step = 0
        episode_time = 0.0

        while not (done or truncated):
            if is_multi and env.current_player != env.agent_player:
                mask = env.get_action_mask()
                action = env.action_space.sample(mask=mask)
                _, reward, done, truncated, _ = env.step(action)
                n_step += 1
            else:
                t0 = time.perf_counter()
                action = monte_carlo_random_rollout(env, num_simulations)
                episode_time += time.perf_counter() - t0

                _, reward, done, truncated, _ = env.step(action)
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

    plot_dir = f"{settings.training_logs_dir}/mcrr/{env.unwrapped}/seed_{seed}/eval"
    os.makedirs(plot_dir, exist_ok=True)

    plot_metric(values=rewards_history, save_dir=plot_dir, window_size=0,
                exp_name=f"{agent_name}_env_{env.unwrapped}", metric_name="winrate")
    plot_metric(values=n_steps_history, save_dir=plot_dir, window_size=0,
                exp_name=f"{agent_name}_env_{env.unwrapped}", metric_name="nbr_steps")
    plot_metric(values=time_per_move_history, save_dir=plot_dir, window_size=0,
                exp_name=f"{agent_name}_env_{env.unwrapped}", metric_name="time_per_move")

    results = {
        "env": str(env.unwrapped),
        "agent": agent_name,
        "seed": seed,
        "num_episodes": num_episodes,
        "num_simulations": num_simulations,
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

    json_path = f"{plot_dir}/{agent_name}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    s = results["summary"]
    print(
        f"[EVAL {env.unwrapped}/{agent_name} | seed={seed}] "
        f"win={s['win_rate']:.2%} loss={s['loss_rate']:.2%} draw={s['draw_rate']:.2%} "
        f"mean_steps={s['mean_steps']:.2f} -> {json_path}"
    )
    return results


def eval_all_for_env(env, seed, num_simulations=100):
    eval_agent(env, num_episodes=1_000, num_simulations=num_simulations, seed=seed)


def wrap_video(env, seed, episode_num_trigger):
    video_env = RecordVideo(
        env,
        video_folder=f"{settings.videos_dir}/mcrr/{env.unwrapped}/seed_{seed}/eval/",
        episode_trigger=lambda ep: ep % episode_num_trigger == 0,
    )
    video_env.state_id = env.state_id
    video_env.get_action_mask = env.get_action_mask
    video_env.determinize = env.determinize
    video_env.agent_player = env.agent_player
    if hasattr(env, "is_multi_player"):
        video_env.is_multi_player = env.is_multi_player
    type(video_env).current_player = property(
        lambda self: env.current_player,
        lambda self, v: setattr(env, "current_player", v)
    )
    return video_env


if __name__ == "__main__":
    env_classes = [LineWorldEnv, GridWorldEnv, TicTacToeEnv, QuartoEnv]

    for seed in SEEDS:
        for EnvCls in env_classes:
            env_eval = EnvCls(render_mode="rgb_array")
            video_env_eval = wrap_video(env_eval, seed, episode_num_trigger=100)
            print(f"\n{'=' * 60}\nEVAL {env_eval.unwrapped} | seed={seed}\n{'=' * 60}")
            eval_all_for_env(video_env_eval, seed, num_simulations=100)
            video_env_eval.close()
