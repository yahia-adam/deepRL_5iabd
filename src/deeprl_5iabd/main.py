from deeprl_5iabd.config import settings
import time
import click
import importlib
from gymnasium.wrappers import RecordVideo

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
@click.option("--record", "-r", is_flag=True, default=False, help="Enregistrer des épisodes en vidéo.")
@click.option("--record-every", default=None, type=int, help="Enregistrer tous les N épisodes. Par défaut: 10 vidéos réparties sur le training.")
def train(env: str, algo: str, episodes: int, record: bool, record_every: int | None):
    """Entraînement : ENV ALGO [--episodes N] [--record] [--record-every N]."""
    if record:
        every = record_every or max(1, episodes // 10)
        _environment = load_env(env, render_mode="rgb_array")
        environment = RecordVideo(
            _environment,
            video_folder=f"{settings.videos_dir}/{env}/{algo}",
            episode_trigger=lambda ep: ep % every == 0,
        )
        environment.state_id = _environment.state_id
        environment.get_action_mask = _environment.get_action_mask
        click.echo(f"Recording every {every} episodes → {settings.videos_dir}/{env}/{algo}")
    else:
        environment = load_env(env)
 
    fn = load_algo(algo)
    fn(environment, num_episodes=episodes)
    environment.close()

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