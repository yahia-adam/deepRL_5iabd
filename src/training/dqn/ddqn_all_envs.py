from deeprl_5iabd.agents.ddqn import ddqn
from deeprl_5iabd.envs.line_world import LineWorldEnv
from deeprl_5iabd.envs.grid_world import GridWorldEnv
from deeprl_5iabd.envs.tictactoe import TicTacToeEnv
from deeprl_5iabd.envs.quarto import QuartoEnv


CONFIGS = [
    # ε décroît linéairement de 1.0 à epsilon_end sur les
    # epsilon_anneal_frac premiers % d'épisodes 
    # Optimiseur RMSProp(momentum=0.95) 
    (LineWorldEnv, dict(num_episodes=20_000, lr=2.5e-4, hidden_size=16,
                        epsilon_anneal_frac=0.05, epsilon_end=0.02, gamma=0.9)),
    (GridWorldEnv, dict(num_episodes=20_000, lr=2.5e-4, hidden_size=32,
                        epsilon_anneal_frac=0.3, epsilon_end=0.1)),
    (TicTacToeEnv, dict(num_episodes=20_000, lr=2.5e-4, hidden_size=128,
                        epsilon_anneal_frac=0.6, epsilon_end=0.1)),
    (QuartoEnv,    dict(num_episodes=30_000, lr=2.5e-4, hidden_size=256,
                        epsilon_anneal_frac=0.7, epsilon_end=0.1)),
]


def main():
    for EnvCls, kwargs in CONFIGS:
        print(f"\n=== Training Double DQN on {EnvCls.__name__} ===")
        env = EnvCls()
        ddqn(env, **kwargs)


if __name__ == "__main__":
    main()
