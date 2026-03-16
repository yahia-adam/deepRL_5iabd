"""Fixtures partagées pour les tests unitaires et d'intégration."""
import pytest
from deeprl_5iabd.envs.line_world import LineWorld
from deeprl_5iabd.envs.grid_world import GridWorld
from deeprl_5iabd.envs.tictactoe import TicTacToe
from deeprl_5iabd.envs.quarto import QuartoEnv
from deeprl_5iabd.agents.policy_net import PolicyNetwork
from deeprl_5iabd.agents.random_agent import RandomPlayer


@pytest.fixture
def line_world():
    return LineWorld()


@pytest.fixture
def grid_world():
    return GridWorld()


@pytest.fixture
def tictactoe_env():
    return TicTacToe()


@pytest.fixture
def quarto_env():
    return QuartoEnv()


@pytest.fixture
def policy_net():
    return PolicyNetwork(name="test_net", input_size=4, output_size=4, hiddenlayers=[16, 8])


@pytest.fixture
def random_player():
    return RandomPlayer(action_dim=4)
