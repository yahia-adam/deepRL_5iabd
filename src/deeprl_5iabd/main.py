import argparse
import time


ENV_MAP = {
    "quarto":    ("deeprl_5iabd.envs.quarto",     "QuartoEnv"),
    "tictactoe": ("deeprl_5iabd.envs.tictactoe",  "TicTacToeEnv"),
    "gridworld": ("deeprl_5iabd.envs.grid_world",  "GridWorldEnv"),
    "lineworld":  ("deeprl_5iabd.envs.line_world",  "LineWorldEnv"),
}

def load_env(name: str, **kwargs):
    import importlib
    module_path, class_name = ENV_MAP[name]
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(**kwargs)

def count_n_match_time(env_name: str, num_episodes: int = 10_000):
    env = load_env(env_name)
    start_time = time.time()
    for _ in range(num_episodes):
        env.reset()
        done = False
        while not done:
            mask = env._get_action_mask()
            action = env.action_space.sample(mask=mask)
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    elapsed = time.time() - start_time
    print(f"Env                : {env_name}")
    print(f"Temps total        : {elapsed:.3f} secondes")
    print(f"Parties par sec    : {num_episodes / elapsed:.0f}")

def play_human_vs_random(env_name: str):
    env = load_env(env_name, render_mode="human")
    obs, _ = env.reset()
    done = False
    while not done:
        if env.current_player == env.agent_player:
            env.render()
            mask = env._get_action_mask()
            action = env.action_space.sample(mask=mask)
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        else:
            env.render()
            mask = env._get_action_mask()
            action = env._wait_for_human_click(mask)
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

if __name__ == "__main__":
    env_choices = list(ENV_MAP.keys())

    parser = argparse.ArgumentParser(description="DeepRL CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench = subparsers.add_parser("bench", help="Benchmark random vs random")
    bench.add_argument("env", choices=env_choices, help="Environnement à utiliser")
    bench.add_argument("--episodes", type=int, default=10_000, metavar="N",
                       help="Nombre de parties (défaut: 10 000)")

    play = subparsers.add_parser("play", help="Human vs Random")
    play.add_argument("env", choices=env_choices, help="Environnement à utiliser")

    args = parser.parse_args()

    if args.command == "bench":
        count_n_match_time(args.env, args.episodes)
    elif args.command == "play":
        play_human_vs_random(args.env)