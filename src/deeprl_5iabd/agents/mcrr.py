import numpy as np
from torch.distributions import Categorical
from deeprl_5iabd.envs.base_env import BaseEnv
from deeprl_5iabd.agents.random_agent import RandomPlayer

def monte_carlo_random_rollout(env: BaseEnv, random_agent: RandomPlayer, num_rollouts: int):
    actions_mask = np.array(env.get_action_space())
    action_mean_rewards = np.full(len(actions_mask), -np.inf)
    actions = np.where(actions_mask == 1)[0]
    action_mean_rewards[actions] = 0

    a_resource = num_rollouts // len(actions)
    for a in actions:
        for _ in range(a_resource):
            new_env = env.determinize()
            new_env.step(a)
            while not new_env.is_game_over():
                probs = random_agent.forward(x=None, mask=new_env.get_action_space())
                probs_dist = Categorical(probs)
                action_pos = probs_dist.sample()
                new_env.step(action_pos.item())
            action_mean_rewards[a] += new_env.score()
        action_mean_rewards[a] //= a_resource 

    best_action_idx = np.argmax(action_mean_rewards)

    return int(actions_mask[best_action_idx])