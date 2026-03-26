import numpy as np
import torch
from torch.distributions import Categorical
from deeprl_5iabd.agents.random_agent import RandomPlayer
from deeprl_5iabd.envs.model_based_env import ModelBasedEnv
from deeprl_5iabd.helper import softmax_with_mask

def choose_action_epsilon_greedy(state, mask, Q, epsilon):
    available = [a for a, m in enumerate(mask) if m == 1]

    if np.random.random() < epsilon:
        return int(np.random.choice(available))
    else:
        q_tensor = torch.tensor(Q[state, :], dtype=torch.float32)
        probs = softmax_with_mask(q_tensor, mask)
        return int(torch.argmax(probs).item())

def q_learning(
    env,
    learning_rate=0.1,
    epsilon=0.1,
    gamma=0.9,
    num_episodes=100_000,
    is_two_players=False
):
    Q = np.zeros((env.num_states(), env.num_actions()))
    rp = RandomPlayer(action_dim=env.num_actions()) if is_two_players else None

    for i in range(num_episodes):
        env.reset()
        s = env.state_id(env.get_observation_space())

        while not env.is_game_over():
            mask = env.get_action_space()
            a = choose_action_epsilon_greedy(s, mask, Q, epsilon)
            
            old_score = env.score()
            env.step(a)

            if is_two_players and not env.is_game_over():
                mask_adv = env.get_action_space()
                probs = rp.forward(x=None, mask=mask_adv)
                m = Categorical(probs)
                a_adv = m.sample().item()
                env.step(a_adv)

            s_prime = env.state_id(env.get_observation_space())
            r = float(env.score()) if is_two_players else float(env.score() - old_score)

            if not env.is_game_over():
                mask_prime = env.get_action_space()
                q_next = Q[s_prime].copy()
                q_next[mask_prime == 0] = -np.inf
                max_q_next = np.max(q_next)
            else:
                max_q_next = 0.0

            Q[s, a] += learning_rate * (r + gamma * max_q_next - Q[s, a])
            s = s_prime

    return Q