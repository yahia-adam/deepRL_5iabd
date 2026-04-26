from deeprl_5iabd.agents.dqn import dqn
from deeprl_5iabd.agents.ddqn import ddqn
from deeprl_5iabd.agents.ddqn_replay import ddqn_replay
from deeprl_5iabd.agents.ddqn_per import ddqn_per
from deeprl_5iabd.envs.tictactoe import TicTacToeEnv


NUM_EPISODES = 20_000
HIDDEN_SIZE = 128


RUNS = [
    # (label, fonction d'entraînement, kwargs)
    ("DQN (online, sans replay)", dqn, dict(
        num_episodes=NUM_EPISODES,
        lr=2e-2,
        hidden_size=HIDDEN_SIZE,
        epsilon_decay=0.9998,
    )),
    ("Double DQN (online, target net)", ddqn, dict(
        num_episodes=NUM_EPISODES,
        lr=2.5e-4,
        hidden_size=HIDDEN_SIZE,
        epsilon_anneal_frac=0.6,
        epsilon_end=0.1,
    )),
    ("DDQN + Replay Buffer", ddqn_replay, dict(
        num_episodes=NUM_EPISODES,
        lr=2.5e-4,
        hidden_size=HIDDEN_SIZE,
        epsilon_anneal_frac=0.5,
        epsilon_end=0.1,
        buffer_capacity=50_000,
        batch_size=32,
        learning_starts=1_000,
        target_update_freq=500,
    )),
    ("DDQN + Prioritized Experience Replay", ddqn_per, dict(
        num_episodes=NUM_EPISODES,
        lr=6.25e-5,
        hidden_size=HIDDEN_SIZE,
        epsilon_anneal_frac=0.5,
        epsilon_end=0.1,
        buffer_capacity=50_000,
        batch_size=32,
        learning_starts=1_000,
        target_update_freq=500,
        alpha=0.6,
        beta_start=0.4,
        beta_end=1.0,
    )),
]


def main():
    for label, train_fn, kwargs in RUNS:
        print(f"\n=== {label} on TicTacToe ({NUM_EPISODES} épisodes) ===")
        env = TicTacToeEnv()
        train_fn(env, **kwargs)


if __name__ == "__main__":
    main()
