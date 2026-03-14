from mypythonlib.config import settings
from mypythonlib.envs.quarto import QuartoEnv
from mypythonlib.agents.reinforce import reinforce
from mypythonlib.agents.base_agent import MyModel
from mypythonlib.agents.random_agent import RandomPlayer
from mypythonlib.tracking.tb_logger import TensorBoardLogger

if __name__ == "__main__":
    env = QuartoEnv()
    model  = MyModel(input_size=len(env.get_observation_space()), output_size=len(env.get_action_space()))
    opponent_model = RandomPlayer(action_dim=len(env.get_action_space()))
    logger = TensorBoardLogger(
        log_dir=settings.training_logs_path, 
        experiment_name="Quarto_REINFORCE_vs_Random"
    )

    model = reinforce(env=env, opponent_model=opponent_model, reinforce_agent=model, logger=logger, num_episodes=2000)
