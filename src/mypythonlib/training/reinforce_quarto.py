from mypythonlib.config import settings
from mypythonlib.envs.quarto import QuartoEnv
from mypythonlib.agents.reinforce import reinforce
from mypythonlib.agents.agent_base import MyModel
from mypythonlib.agents.random_agent import RandomPlayer

if __name__ == "__main__":
    env = QuartoEnv()
    model  = MyModel(input_size=len(env.get_observation_space()), output_size=len(env.get_action_space()))
    oponent_model = RandomPlayer(len(env.get_action_space()))
    model = reinforce(env=env, oponent_model=oponent_model, reinforce_agent=model)
