import numpy as np
from deeprl_5iabd.config import settings
from deeprl_5iabd.envs.line_world import LineWorld
from deeprl_5iabd.agents.q_learning import q_learning

if __name__ == "__main__":
    env = LineWorld()
    Q = q_learning(env=env)

    print(Q)
    action_names = ["Gauche", "Droite"]
    terminal = {0: "mur gauche (punition)", env.num_states()-1: "mur droit (récompense)"}

    print("\n=== Q-Table : score espéré par état et action ===")
    print("Plus la valeur est haute, plus l'action est prometteuse.\n")

    for s in range(env.num_states()):
        if s in terminal:
            print(f"Etat {s} — {terminal[s]} : fin de partie, aucune action possible")
        else:
            best = int(np.argmax(Q[s]))
            print(f"Etat {s} — position {s} sur 4 :")
            for a, name in enumerate(action_names):
                tag = " <-- meilleur choix" if a == best else ""
                print(f"   {name} : {Q[s,a]:.2f}{tag}")
        print()