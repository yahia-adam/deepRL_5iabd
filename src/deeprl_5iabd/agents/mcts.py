import time
import os
import numpy as np
from gymnasium import Env
import matplotlib.pyplot as plt
from collections import deque

from deeprl_5iabd.envs.line_world import LineWorldEnv
from deeprl_5iabd.envs.grid_world import GridWorldEnv
from deeprl_5iabd.envs.tictactoe import TicTacToeEnv
from deeprl_5iabd.envs.quarto import QuartoEnv
from deeprl_5iabd.config import settings
import math


import numpy as np
import math


class MCTSNode:
    def __init__(self, env, parent=None, action=None):
        self.env = env
        self.parent = parent
        self.action = action

        self.children = []
        self.visits = 0
        self.value = 0.0

        self.untried_actions = np.where(env.get_action_mask() == 1)[0]


    # SELECTION
    def selection(self, c_param=1.4):
        node = self

        while node.is_fully_expanded() and node.children:
            node = node.best_child(c_param)

        return node


    def is_fully_expanded(self):
        return len(self.untried_actions) == 0


    def best_child(self, c_param=1.4):

        is_agent_turn = (self.env.current_player == self.env.agent_player)

        choices = []

        for child in self.children:

            exploit = child.value / child.visits

            if not is_agent_turn:
                exploit = -exploit

            explore = c_param * math.sqrt(
                math.log(self.visits + 1) / child.visits
            )

            choices.append(exploit + explore)

        return self.children[np.argmax(choices)]


    # EXPANSION
    def expansion(self):
        action = self.untried_actions[0]
        self.untried_actions = self.untried_actions[1:]

        new_env = self.env.determinize()
        new_env.step(action)

        action_node = MCTSNode(new_env, parent=self, action=action)
        self.children.append(action_node)

        return action_node


    # SIMULATION
    def simulation(self):
        env = self.env.determinize()

        terminated = False
        truncated = False

        total_reward = 0

        while not (terminated or truncated):
            mask = env.get_action_mask()
            action = env.action_space.sample(mask=mask)

            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

        return total_reward


    # BACKPROPAGATION
    def backpropagate(self, reward):
        self.visits += 1
        self.value += reward

        if self.parent:
            self.parent.backpropagate(reward)


def mcts(env, num_simulations=100):

    root = MCTSNode(env)

    for _ in range(num_simulations):

        # 1. SELECTION
        node = root.selection()

        # 2. EXPANSION
        if not node.is_fully_expanded():
            node = node.expansion()

        # 3. SIMULATION
        reward = node.simulation()

        # 4. BACKPROP
        node.backpropagate(reward)

    # action finale = la plus visitée
    best_node = max(root.children, key=lambda c: c.visits)

    return best_node.action


def run_mcts(env: Env, num_episodes: int, num_simulations: int):
    rewards = []
    wins_rate, draws_rate, losses_rate = [], [], []

    for episode in range(num_episodes):
        done = False
        env.reset()
        final_reward = 0

        if not env.is_multi_player:
            while not done:
                a = mcts(env, num_simulations)
                _, reward, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                final_reward = reward
        else:
            while not done:
                if env.current_player == env.agent_player:
                    a = mcts(env, num_simulations)
                else:
                    mask = env.get_action_mask()
                    a = env.action_space.sample(mask=mask)

                _, reward, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                final_reward = reward

        rewards.append(final_reward)

        # --- Stats tous les 100 épisodes ---
        if (episode + 1) % 100 == 0:
            last_100 = np.array(rewards[-100:])
            wins   = np.sum(last_100 ==  1)
            draws  = np.sum(last_100 ==  0)
            losses = np.sum(last_100 == -1)

            wins_rate.append(wins)
            draws_rate.append(draws)
            losses_rate.append(losses)

            print(f"[{episode+1}/{num_episodes}] Win: {wins}% | Draw: {draws}% | Loss: {losses}%")

    # --- Plot ---
    x = np.arange(100, num_episodes + 1, 100)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, wins_rate,   label="Victoires %", color="green")
    ax.plot(x, draws_rate,  label="Nuls %",      color="orange")
    ax.plot(x, losses_rate, label="Défaites %",  color="red")
    ax.set_xlabel("Épisode")
    ax.set_ylabel("% sur 100 épisodes glissants")
    ax.set_title(f"MCTS — {env.unwrapped} | simulations={num_simulations}")
    ax.legend()
    plt.tight_layout()

    save_dir = f"{settings.training_logs_path}/MCTS/{env.unwrapped}"
    os.makedirs(save_dir, exist_ok=True)
    path = f"{save_dir}/mcts_{env.unwrapped}num_episodes{num_episodes}num_simulations{num_simulations}.png"
    plt.savefig(path)
    plt.close()

    print(f"Plot saved → {path}")


if __name__ == "__main__":
    # env = LineWorldEnv()
    # run_mcts(env, num_episodes=1_000, num_simulations=100)

    # env = GridWorldEnv()
    # run_mcts(env, num_episodes=1_000, num_simulations=50)

    env = TicTacToeEnv()
    run_mcts(env, num_episodes=1_000, num_simulations=10)

    # env = QuartoEnv()
    # run_mcts(env, num_episodes=1_000, num_simulations=100)
