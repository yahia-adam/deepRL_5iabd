

from src.agents.q_learning import q_learning
from src.envs.line_world import LineWorld


def main():
    env = LineWorld()
    Q = q_learning(env)
    print(Q)

if __name__ == "__main__":
    main()