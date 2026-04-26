from deeprl_5iabd.agents.ddqn_per import ddqn_per
from deeprl_5iabd.envs.line_world import LineWorldEnv
from deeprl_5iabd.envs.grid_world import GridWorldEnv
from deeprl_5iabd.envs.tictactoe import TicTacToeEnv
from deeprl_5iabd.envs.quarto import QuartoEnv


CONFIGS = [
    # epsilon décroît linéairement, RMSProp(momentum=0.95), batch=32, train_freq=4,
    # lr = lr_DDQN / 4 (papier PER : "we reduced the step-size η by a factor 4").
    (LineWorldEnv, dict(
        num_episodes=1_000, lr=6.25e-5, hidden_size=16,
        epsilon_anneal_frac=0.1, epsilon_end=0.02, gamma=0.9,
        buffer_capacity=5_000, batch_size=32, learning_starts=200,
        target_update_freq=200,
        alpha=0.6, beta_start=0.4, beta_end=1.0,
    )),
    (GridWorldEnv, dict(
        num_episodes=5_000, lr=6.25e-5, hidden_size=32,
        epsilon_anneal_frac=0.3, epsilon_end=0.1,
        buffer_capacity=20_000, batch_size=32, learning_starts=500,
        target_update_freq=500,
        alpha=0.6, beta_start=0.4, beta_end=1.0,
    )),
    (TicTacToeEnv, dict(
        num_episodes=10_000, lr=6.25e-5, hidden_size=128,
        epsilon_anneal_frac=0.5, epsilon_end=0.1,
        buffer_capacity=50_000, batch_size=32, learning_starts=1_000,
        target_update_freq=500,
        alpha=0.6, beta_start=0.4, beta_end=1.0,
    )),
    (QuartoEnv, dict(
        num_episodes=20_000, lr=6.25e-5, hidden_size=256,
        epsilon_anneal_frac=0.6, epsilon_end=0.1,
        buffer_capacity=100_000, batch_size=32, learning_starts=2_000,
        target_update_freq=1_000,
        alpha=0.6, beta_start=0.4, beta_end=1.0,
    )),
]


def main():
    for EnvCls, kwargs in CONFIGS:
        print(f"\n=== Training DDQN + PER on {EnvCls.__name__} ===")
        env = EnvCls()
        ddqn_per(env, **kwargs)


if __name__ == "__main__":
    main()
