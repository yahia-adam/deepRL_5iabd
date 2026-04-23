from deeprl_5iabd.agents.dqn import dqn
from deeprl_5iabd.envs.line_world import LineWorldEnv
from deeprl_5iabd.envs.grid_world import GridWorldEnv
from deeprl_5iabd.envs.tictactoe import TicTacToeEnv
from deeprl_5iabd.envs.quarto import QuartoEnv

CONFIGS = [
    (LineWorldEnv, dict(num_episodes=3_000,  lr=1e-3, hidden_size=64)),
    (GridWorldEnv, dict(num_episodes=10_000, lr=1e-3, hidden_size=128)),
    (TicTacToeEnv, dict(num_episodes=20_000, lr=5e-4, hidden_size=128, epsilon_decay=0.9998)),
    (QuartoEnv,    dict(num_episodes=30_000, lr=5e-4, hidden_size=256, epsilon_decay=0.9999)),
]

def main():
    for EnvCls, kwargs in CONFIGS:
        print(f"\n=== Training DQN on {EnvCls.__name__} ===")
        env = EnvCls()
        dqn(env, **kwargs)

if __name__ == "__main__":
    main()