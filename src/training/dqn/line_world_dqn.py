import numpy as np
import torch
from deeprl_5iabd.agents.dqn import dqn
from deeprl_5iabd.envs.line_world import LineWorld


def main():
    env = LineWorld()

    online_net = dqn(
        env,
        num_episodes=50_000, hidden_size=32, learning_rate=0.01,
        gamma=0.9, batch_size=32, buffer_capacity=5_000,
    )

    print("Q-values apprises (DQN) :")
    print(f"{'État':<8} {'Action 0 (←)':>14} {'Action 1 (→)':>14}")
    print("-" * 38)
    terminal = {0, 4}
    for s in range(env.BOARD_SIZE):
        obs = np.array([s], dtype=np.float32)
        x = torch.tensor(obs).unsqueeze(0)
        with torch.no_grad():
            q = online_net(x)[0].numpy()
        marker = " ← terminal" if s in terminal else ""
        print(f"  s={s}   {q[0]:>12.4f}   {q[1]:>12.4f}{marker}")


if __name__ == "__main__":
    main()
