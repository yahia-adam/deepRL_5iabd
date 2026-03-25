import numpy as np
import torch
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
    env: ModelBasedEnv,
    learning_rate = 0.1,
    epsilon = 0.1,
    gamma = 0.9,
    num_episodes = 100_000
):
    Q = np.random.randn(env.num_states(), env.num_actions())

    for terminal_state in env.T:
        Q[env.state_id(terminal_state), :] = 0

    for _ in range(num_episodes):
        env.reset()
        s = env.state_id(env.get_observation_space())

        while not env.is_game_over():
            a = choose_action_epsilon_greedy(s, env.get_action_space(), Q, epsilon)

            old_score = env.score()
            env.step(a)

            s_prime = env.state_id(env.get_observation_space())
            r = env.score() - old_score

            if not env.is_game_over():
                mask_prime = env.get_action_space()
                q_next_tensor = torch.tensor(Q[s_prime], dtype=torch.float32)
                probs_next = softmax_with_mask(q_next_tensor, mask_prime)
                best_action_next = torch.argmax(probs_next).item()
                max_q_next = Q[s_prime, best_action_next]
            else:
                max_q_next = 0

            Q[s, a] += learning_rate * (r + gamma * max_q_next - Q[s, a])
            s = s_prime

    return Q