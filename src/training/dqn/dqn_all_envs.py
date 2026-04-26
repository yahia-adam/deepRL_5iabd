from deeprl_5iabd.agents.dqn import dqn
from deeprl_5iabd.envs.line_world import LineWorldEnv
from deeprl_5iabd.envs.grid_world import GridWorldEnv
from deeprl_5iabd.envs.tictactoe import TicTacToeEnv
from deeprl_5iabd.envs.quarto import QuartoEnv

CONFIGS = [
    # Optimiseur RMSProp(momentum=0.95) appliqué dans dqn.py 
    # lr ~2.5e-4 conforme au papier ; valeurs alignées sur les configs DDQN
    # car l'optimiseur est désormais le même.
    # Petits environnements : réseau minimal
    (LineWorldEnv, dict(num_episodes=1_500,  lr=2.5e-4, hidden_size=16,
                        epsilon_decay=0.995, epsilon_end=0.02, gamma=0.9)),
    (GridWorldEnv, dict(num_episodes=20_000, lr=2.5e-4, hidden_size=32,
                        epsilon_decay=0.9995, epsilon_end=0.1)),
    # Environnements plus complexes : réseau plus grand
    (TicTacToeEnv, dict(num_episodes=20_000, lr=2.5e-4, hidden_size=128,
                        epsilon_decay=0.9998)),
    (QuartoEnv,    dict(num_episodes=30_000, lr=2.5e-4, hidden_size=256,
                        epsilon_decay=0.9999)),
]

def main():
    for EnvCls, kwargs in CONFIGS:
        print(f"\n=== Training DQN on {EnvCls.__name__} ===")
        env = EnvCls()
        dqn(env, **kwargs)

if __name__ == "__main__":
    main()