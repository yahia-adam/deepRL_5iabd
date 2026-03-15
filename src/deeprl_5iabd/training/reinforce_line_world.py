from deeprl_5iabd.config import settings
from deeprl_5iabd.envs.line_world import LineWorld
from deeprl_5iabd.agents.reinforce import reinforce
from deeprl_5iabd.agents.policy_net import PolicyNetwork
from deeprl_5iabd.agents.random_agent import RandomPlayer
from deeprl_5iabd.tracking.tb_logger import TensorBoardLogger

if __name__ == "__main__":
    env = LineWorld()
    model  = PolicyNetwork(name="LineWorld_REINFORCE", input_size=len(env.get_observation_space()), output_size=len(env.get_action_space()))
    logger = TensorBoardLogger(
        log_dir=settings.training_logs_path, 
        experiment_name="LineWorld_REINFORCE_vs_Random"
    )

    model = reinforce(env=env, opponent_model=None, reinforce_agent=model, logger=logger, num_episodes=2000)
