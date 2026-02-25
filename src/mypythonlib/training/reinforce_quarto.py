from mypythonlib.config import settings
from mypythonlib.envs.quarto import QuartoEnv
from mypythonlib.agents.reinforce import reinforce

if __name__ == "__main__":
    env = QuartoEnv()
    model = reinforce(env=env)
