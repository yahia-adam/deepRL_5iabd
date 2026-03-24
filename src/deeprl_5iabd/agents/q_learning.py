from deeprl_5iabd.envs.model_based_env import ModelBasedEnv
import numpy as np

def choose_action_epsilon_greedy(state, available_actions, Q, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice(available_actions)
    else:
        q_values = Q[state, available_actions]
        best_action_index = np.argmax(q_values)
        return available_actions[best_action_index]

def q_learning(
    env: ModelBasedEnv,
    learning_rate = 0.01,      # ou lr
    epsilon = 0.1,            # pour epsilon-greedy
    gamma = 0.9,              # facteur d'attenuation
    num_episodes = 100_000       # nombre d'épisodes à entraîner
):
    Q = np.random.randn(env.num_states(), env.num_actions())

    for terminal_state in env.T:
        Q[env.state_id(terminal_state), :] = 0

    for _ in range(num_episodes):
        env.reset()
        s = env.state_id(env.get_observation_space())

        while not env.is_game_over():
            a = choose_action_epsilon_greedy(s, env.available_actions(), Q, epsilon)
            env.step(a)
            s_prime = env.state_id(env.get_observation_space())

            r = env.score() 

            if not env.is_game_over():
                available_actions_prime = env.available_actions()
                q_values_prime = Q[s_prime, available_actions_prime]
                max_q_next = np.max(q_values_prime)
            else:
                max_q_next = 0

            Q[s,a] += learning_rate * (r + gamma * max_q_next - Q[s, a])
            s = s_prime
    return Q
