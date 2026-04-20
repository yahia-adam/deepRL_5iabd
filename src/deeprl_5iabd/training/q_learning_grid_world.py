import numpy as np
from deeprl_5iabd.config import settings
from deeprl_5iabd.envs.grid_world import GridWorld
from deeprl_5iabd.agents.q_learning import q_learning

if __name__ == "__main__":
    print("test1")
    print("test1")
    print("test1")
    print("test1")
    print("test1")



    env = GridWorld()
    Q = q_learning(env=env)

    action_arrows = {0: "↓", 1: "↑", 2: "→", 3: "←"}
    goal  = (0, 4)
    trap  = (4, 4)
    size  = 5

    print("\n=== Politique apprise (meilleure action par case) ===\n")

    for row in range(size):
        line = ""
        for col in range(size):
            if (row, col) == goal:
                line += " G "
            elif (row, col) == trap:
                line += " P "
            else:
                sid  = row * size + col
                best = int(np.argmax(Q[sid]))
                line += f" {action_arrows[best]} "
        print(line)

    print("\nG = Goal (+1)   P = Piège (-1)")