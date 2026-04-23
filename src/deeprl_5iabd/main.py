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

@cli.command()
@click.argument("env", type=ENV_CHOICES)
@click.option("--mode", type=click.Choice(["hvr", "rvm", "mvm"]), default="hvr",
              help="hvr=Human vs Random, rvm=Random vs Model, mvm=Model vs Model")
@click.option("--model", "model_path", type=click.Path(exists=True), default=None,
              help="Path to .pkl model (required for rvm/mvm)")
@click.option("--model2", "model2_path", type=click.Path(exists=True), default=None,
              help="Path to second .pkl model (optional for mvm, else same as --model)")
@click.option("--delay", type=float, default=0.5,
              help="Delay (s) between AI moves for visibility (default 0.5)")
def play(env: str, mode: str, model_path: str | None, model2_path: str | None, delay: float):
    """Parties en boucle avec rendu graphique. Ctrl+C pour quitter."""

    def load_model(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    if mode in ("rvm", "mvm") and not model_path:
        raise click.UsageError(f"--model is required for mode '{mode}'")
    if mode == "mvm" and not model2_path:
        model2_path = model_path

    model = load_model(model_path) if model_path else None
    model2 = load_model(model2_path) if model2_path else None

    environment = load_env(env, render_mode="human")

    if not environment.is_multi_player:
        solo_agent = "human" if mode == "hvr" else model
    else:
        agents = {
            "hvr": ("human", "random"),
            "rvm": ("random", model),
            "mvm": (model, model2),
        }
        agent_p0, agent_p1 = agents[mode]

    def pick_action(agent, mask):
        if agent == "human":
            return environment._wait_for_human_click(mask)
        if agent == "random":
            time.sleep(delay)
            return environment.action_space.sample(mask=mask)
        time.sleep(delay)
        return agent.choose_action(environment, mask)

    while True:
        environment.reset()
        done = False

        while not done:
            environment.render()
            mask = environment.get_action_mask()

            if environment.is_multi_player:
                agent = agent_p1 if environment.current_player == environment.agent_player else agent_p0
            else:
                agent = solo_agent

            action = pick_action(agent, mask)
            _, reward, terminated, truncated, _ = environment.step(action)
            done = terminated or truncated
            print(f"reward: {reward}")

        # Pause entre les parties pour voir le résultat final
        environment.render()
        time.sleep(delay * 3)
    
if __name__ == "__main__":
    cli()