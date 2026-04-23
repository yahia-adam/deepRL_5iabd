import time
import click
import importlib


ENV_MAP = {
    "quarto":    ("deeprl_5iabd.envs.quarto",     "QuartoEnv"),
    "tictactoe": ("deeprl_5iabd.envs.tictactoe",  "TicTacToeEnv"),
    "gridworld": ("deeprl_5iabd.envs.grid_world",  "GridWorldEnv"),
    "lineworld":  ("deeprl_5iabd.envs.line_world",  "LineWorldEnv"),
}


ALGO_MAP = {
    "q_learning":  ("deeprl_5iabd.agents.q_learning",  "q_learning"),
    "q_learning_tictactoe": ("deeprl_5iabd.agents.q_learning", "q_learning_tictactoe"),
    "reinforce":   ("deeprl_5iabd.agents.reinforce",   "reinforce"),
}


ENV_CHOICES = click.Choice(list(ENV_MAP.keys()))
ALGO_CHOICES = click.Choice(list(ALGO_MAP.keys()))


def load_env(name: str, **kwargs):
    module_path, class_name = ENV_MAP[name]
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(**kwargs)


@click.group()
def cli():
    """DeepRL CLI — entraînement et benchmark d'agents RL."""


@cli.command()
@click.argument("env", type=ENV_CHOICES)
@click.option("--episodes", "-n", default=10_000, show_default=True, help="Nombre de parties.")
def bench(env: str, episodes: int):
    """Benchmark random vs random."""
    environment = load_env(env)
    start = time.time()
    for _ in range(episodes):
        environment.reset()
        done = False
        while not done:
            mask = environment.get_action_mask()
            action = environment.action_space.sample(mask=mask)
            _, _, terminated, truncated, _ = environment.step(action)
            done = terminated or truncated
    elapsed = time.time() - start
    click.echo(f"Env             : {env}")
    click.echo(f"Temps total     : {elapsed:.3f} secondes")
    click.echo(f"Parties par sec : {episodes / elapsed:.0f}")


def load_algo(name: str):
    module_path, fn_name = ALGO_MAP[name]
    return getattr(importlib.import_module(module_path), fn_name)


@cli.command()
@click.argument("env",  type=ENV_CHOICES)
@click.argument("algo", type=ALGO_CHOICES)
@click.option("--episodes", "-n", default=10_000, show_default=True, help="Nombre d'épisodes.")
def train(env: str, algo: str, episodes: int):
    """Entraînement : ENV ALGO [--episodes N]."""
    environment = load_env(env)
    fn = load_algo(algo, )
    fn(environment, num_episodes=episodes)


@cli.command()
@click.argument("env", type=ENV_CHOICES)
def play(env: str):
    """Human vs Random en mode graphique (parties en boucle, Ctrl+C pour quitter)."""
    environment = load_env(env, render_mode="human")
    while True:

        done = False
        environment.reset()
        while not done:
            environment.render()
            mask = environment.get_action_mask()
            print(mask)
    
            if environment.is_multi_player:

                if (environment.current_player == environment.agent_player):
                    action = environment.action_space.sample(mask=mask)
                else:
                    action = environment._wait_for_human_click(mask)

            else:

                action = environment._wait_for_human_click(mask)

            _, reward, terminated, truncated, _ = environment.step(action)
            done = terminated or truncated
            print(f"reward : {reward}")


if __name__ == "__main__":
    cli()