import torch
from mypythonlib.envs.quarto import QuartoEnv
from mypythonlib.agents.reinforce import ReinforceModel

if __name__ == "__main__":
    env = QuartoEnv()
    agent = ReinforceModel(len(env.get_observation_space()), len(env.get_action_space()))

    obs_tensor = torch.tensor(env.get_observation_space()).float().unsqueeze(0)
    mask_tensor = torch.tensor(env.get_action_space()).float().unsqueeze(0)

    print(agent.forward(obs_tensor, mask_tensor))