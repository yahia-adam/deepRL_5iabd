import time
import click
import pickle
import importlib
from gymnasium.wrappers import RecordVideo
from deeprl_5iabd.config import settings

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


MAIN_EXAMPLES = """
\b
Examples:
  main.py bench quarto --episodes 5000
  main.py train tictactoe q_learning --episodes 20000 --record
  main.py train lineworld reinforce -n 50000 --record --record-every 1000
  main.py play quarto
"""

@click.group(epilog=MAIN_EXAMPLES)
def cli():
    """DeepRL CLI — entraînement et benchmark d'agents RL."""


BENCH_EXAMPLES = """
\b
Examples:
  main.py bench quarto
  main.py bench tictactoe --episodes 5000
  main.py bench lineworld -n 50000
"""

@cli.command(epilog=BENCH_EXAMPLES)
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


TRAIN_EXAMPLES = """
\b
Examples:
  main.py train tictactoe q_learning
  main.py train quarto reinforce --episodes 50000
  main.py train lineworld q_learning -n 20000 --record
  main.py train gridworld reinforce -n 100000 --record --record-every 5000
"""

@cli.command(epilog=TRAIN_EXAMPLES)
@click.argument("env",  type=ENV_CHOICES)
@click.argument("algo", type=ALGO_CHOICES)
@click.option("--episodes", "-n", default=10_000, show_default=True, help="Nombre d'épisodes.")
@click.option("--record", "-r", is_flag=True, default=False, help="Enregistrer des épisodes en vidéo.")
@click.option("--record-every", default=None, type=int, help="Enregistrer tous les N épisodes. Par défaut: 10 vidéos réparties sur le training.")
def train(env: str, algo: str, episodes: int, record: bool, record_every: int | None):
    """Entraînement : ENV ALGO [--episodes N] [--record] [--record-every N]."""
    if record:
        every = record_every or max(1, episodes // 10)
        _environment = load_env(env, render_mode="rgb_array")
        environment = RecordVideo(
            _environment,
            video_folder=f"{settings.videos_dir}/{algo}/{env}",
            episode_trigger=lambda ep: ep % every == 0,
        )
        environment.state_id = _environment.state_id
        environment.get_action_mask = _environment.get_action_mask
        environment.agent_player = _environment.agent_player
        type(environment).current_player = property(
            lambda self: _environment.current_player,
            lambda self, v: setattr(_environment, 'current_player', v)
        )
        click.echo(f"Recording every {every} episodes → {settings.videos_dir}/{algo}/{env}")
    else:
        environment = load_env(env)

    if algo == "q_learning" and env == "tictactoe":
        fn = load_algo("q_learning_tictactoe")
    else:
        fn = load_algo(algo)

    fn(environment, num_episodes=episodes)
    environment.close()


PLAY_EXAMPLES = """
\b
Examples:
  main.py play quarto
  main.py play tictactoe
  main.py play gridworld
"""

@cli.command(epilog=PLAY_EXAMPLES)
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