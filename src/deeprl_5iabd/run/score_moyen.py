import numpy as np

from deeprl_5iabd.envs.line_world import LineWorldEnv
from deeprl_5iabd.envs.grid_world import GridWorldEnv
from deeprl_5iabd.envs.tictactoe import TicTacToeEnv
from deeprl_5iabd.envs.quarto import QuartoEnv

from deeprl_5iabd.agents.reinforce import reinforce
from deeprl_5iabd.agents.ppo import ppo
from deeprl_5iabd.agents.mcrr import monte_carlo_random_rollout
from deeprl_5iabd.agents.mcts import mcts
from deeprl_5iabd.agents.expert_apprenti import mcts




if __name__ == "__main__":
    lineenv = LineWorldEnv()
    gridenv = GridWorldEnv()
    tttenv = TicTacToeEnv()
    quartoenv = QuartoEnv()

