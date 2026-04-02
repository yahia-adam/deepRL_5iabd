import numpy as np
from deeprl_5iabd.envs.grid_world import GridWorld
from deeprl_5iabd.envs.line_world import LineWorld
from deeprl_5iabd.envs.tictactoe import TicTacToe
from deeprl_5iabd.agents.mcrr import monte_carlo_random_rollout
from deeprl_5iabd.agents.random_agent import RandomPlayer

if __name__ == "__main__":
    # env = LineWorld()
    # env = GridWorld()
    env = TicTacToe()

    random_agent = RandomPlayer(action_dim=len(env.get_action_space()))
    rewards = []

    for _ in range(10):
        env.reset()
        while not env.is_game_over():
            best_action = monte_carlo_random_rollout(env, random_agent, 90)
            env.step(best_action)
        s = env.score()
        rewards.append(s)
        print(f"score: {s}")
    print(f"mean score: {np.mean(rewards)}")