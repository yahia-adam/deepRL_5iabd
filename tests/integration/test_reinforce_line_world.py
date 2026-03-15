"""Tests d'intégration : boucle REINFORCE sur LineWorld."""
import pytest
import torch
from deeprl_5iabd.envs.line_world import LineWorld
from deeprl_5iabd.agents.policy_net import PolicyNetwork
from deeprl_5iabd.agents.random_agent import RandomPlayer
from deeprl_5iabd.agents.reinforce import reinforce


@pytest.fixture
def line_env():
    return LineWorld()


@pytest.fixture
def reinforce_agent():
    # LineWorld : obs = [pos] (1 val), actions = [left, right] (2 actions)
    return PolicyNetwork(
        name="reinforce_lw",
        input_size=1,
        output_size=2,
        hiddenlayers=[16, 8]
    )


@pytest.fixture
def opponent():
    # Pas d'adversaire dans LineWorld, mais reinforce() en attend un
    return RandomPlayer(action_dim=2)


class TestReinforceLineWorld:
    def test_runs_without_error(self, line_env, opponent, reinforce_agent):
        """La boucle REINFORCE doit s'exécuter sans lever d'exception."""
        result = reinforce(
            env=line_env,
            opponent_model=opponent,
            reinforce_agent=reinforce_agent,
            logger=None,
            num_episodes=10,
        )
        assert result is not None

    def test_returns_policy_network(self, line_env, opponent, reinforce_agent):
        """reinforce() doit retourner un objet PolicyNetwork."""
        result = reinforce(
            env=line_env,
            opponent_model=opponent,
            reinforce_agent=reinforce_agent,
            logger=None,
            num_episodes=10,
        )
        assert isinstance(result, PolicyNetwork)

    def test_agent_parameters_updated(self, line_env, opponent, reinforce_agent):
        """Les paramètres du réseau doivent avoir été modifiés après entraînement."""
        # Capturer les poids initiaux
        initial_weights = [p.clone() for p in reinforce_agent.parameters()]

        reinforce(
            env=line_env,
            opponent_model=opponent,
            reinforce_agent=reinforce_agent,
            logger=None,
            num_episodes=20,
        )

        # Au moins un paramètre doit avoir changé
        weights_changed = any(
            not torch.equal(p, w)
            for p, w in zip(reinforce_agent.parameters(), initial_weights)
        )
        assert weights_changed, "Les poids du réseau n'ont pas été mis à jour"

    def test_env_resets_between_episodes(self, line_env, opponent, reinforce_agent):
        """L'environnement doit être dans un état valide après l'entraînement."""
        reinforce(
            env=line_env,
            opponent_model=opponent,
            reinforce_agent=reinforce_agent,
            logger=None,
            num_episodes=5,
        )
        # Après l'entraînement, on peut réinitialiser et jouer normalement
        line_env.reset()
        assert not line_env.is_game_over()
        assert line_env.get_observation_space() == [2]
