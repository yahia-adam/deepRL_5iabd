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
from deeprl_5iabd.helper import plot_metric
from gymnasium.wrappers import RecordVideo


def monte_carlo_random_rollout(env: Env, num_simulations: int):

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

    best_action_idx = np.argmax(action_mean_rewards)
    return int(best_action_idx)


def run_mcrr(env: Env, num_episodes: int = 1_000, num_simulations: int = 50):
    print(f"starting mcrr for {env.unwrapped} num_episodes={num_episodes} num_simulations={num_simulations}")

    rewards_history = np.zeros(num_episodes + 1)
    nbr_steps_history = np.zeros(num_episodes + 1)
    game_times_history = np.zeros(num_episodes + 1)

    is_multi = getattr(env, "is_multi_player", False)

    for episode in range(1, num_episodes+1):
        env.reset()
        terminated = False
        truncated = False
        reward = 0.0
        n_step = 0

        t0 = time.perf_counter()
        while not (terminated or truncated):
            if is_multi and env.current_player != env.agent_player:
                mask = env.get_action_mask()
                a = env.action_space.sample(mask=mask)
            else:
                a = monte_carlo_random_rollout(env, num_simulations)

            _, reward, terminated, truncated, _ = env.step(a)
            n_step += 1

        game_time = time.perf_counter() - t0
        rewards_history[episode - 1] = reward
        nbr_steps_history[episode - 1] = n_step
        game_times_history[episode - 1] = game_time

        if episode % 100 == 0:
            print(f"episode {episode} reward={reward} n_step={n_step} game_time={game_time}s")

    save_dir = f"{settings.evaluation_logs_dir}/mcrr/{env.unwrapped}"
    exp_name = f"mcrr_sim{num_simulations}_env_{env.unwrapped}"

    winrate_path = plot_metric(
        values=rewards_history,
        save_dir=save_dir,
        window_size=100,
        exp_name=exp_name,
        metric_name="winrate",
    )
    nbr_steps_path = plot_metric(
        values=nbr_steps_history,
        save_dir=save_dir,
        window_size=100,
        exp_name=exp_name,
        metric_name="nbr_steps",
    )
    game_time_path = plot_metric(
        values=game_times_history,
        save_dir=save_dir,
        window_size=100,
        exp_name=exp_name,
        metric_name="game_time",
    )

    print(f"winrate_path={winrate_path}")
    print(f"nbr_steps_path={nbr_steps_path}")
    print(f"game_time_path={game_time_path}")

def hvr(env, num_episodes, num_simulations):


    for e in range(num_episodes):
        env.reset()
        done = False
        while not done:
            env.render()
            if env.current_player == env.agent_player:
                a = monte_carlo_random_rollout(env, num_simulations)
            else:
                mask = env.get_action_mask()
                a = env._wait_for_human_click(mask)

            _, reward, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
        print(reward)
        env.render()
        time.sleep(5)

    env.close()

if __name__ == "__main__":

    # env = QuartoEnv(render_mode="human")
    # hvr(env, num_episodes=10, num_simulations=200)



    env = LineWorldEnv(render_mode="rgb_array")
    # env = GridWorldEnv(render_mode="rgb_array")
    # env = TicTacToeEnv(render_mode="rgb_array")
    # env = QuartoEnv(render_mode="rgb_array")

    video_env = RecordVideo(
        env,
        video_folder=f"{settings.videos_dir}/mcrr/{env.unwrapped}/eval/",
        episode_trigger=lambda ep: ep % 1_000 == 0,
    )
    video_env.state_id = env.state_id
    video_env.get_action_mask = env.get_action_mask
    video_env.determinize = env.determinize
    video_env.agent_player = env.agent_player
    type(video_env).current_player = property(
        lambda self: env.current_player,
        lambda self, v: setattr(env, 'current_player', v)
    )
    run_mcrr(video_env, num_episodes=10_000, num_simulations=100)
    video_env.close()
