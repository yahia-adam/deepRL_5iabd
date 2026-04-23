from src.envs.base_env import BaseEnv
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


# Simple MLP To estimate the Q-values
class QNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


#  Replay Buffer (Experience Replay)
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_prime, done):
        self.buffer.append((s, a, r, s_prime, done))
    # sample a batch of experiences from the replay buffer
    def sample(self, batch_size: int):
        # selects batch_size(n) batch indices within buffer length
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        # returns a list of selected batch experiences
        batch = [self.buffer[i] for i in indices]
        # Rreturns tuples of each experience elements
        s, a, r, s_prime, done = zip(*batch)
        # Conversion into np.array to allow for tensor operations
        return (
            np.array(s),
            np.array(a),
            np.array(r, dtype=np.float32),
            np.array(s_prime),
            np.array(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# UTILS

def one_hot(state: int, num_states: int) -> np.ndarray:
    """Encode un état entier en vecteur one-hot."""
    v = np.zeros(num_states, dtype=np.float32)
    # On met à 1 la valeur de l'état courant
    v[state] = 1.0
    return v


def choose_action_epsilon_greedy(state: int, available_actions: np.ndarray,
                                  network: QNetwork, num_states: int,
                                  epsilon: float) -> int:
    if np.random.random() < epsilon:
        return int(np.random.choice(available_actions))
    else:
        # unsqueeze to add a dimension to the tensor to make it a 2D tensor for better compatibility with the network
        x = torch.tensor(one_hot(state, num_states)).unsqueeze(0)  # (1, |S|)
        # no_grad to disable gradient tracking for inference
        with torch.no_grad():
              # network(x) to get the q-values for the state
             # [0].numpy() to get the q-values of left and righht actions for the state
            q_values = network(x)[0].numpy()                        # (|A|,)
        q_values_avail = q_values[available_actions]
        best_action_index = np.argmax(q_values_avail)
        return int(available_actions[best_action_index])


#  DQN 
def dqn(
    env: BaseEnv, # The environment to train the agent on deeep 
    learning_rate: float = 0.01, # The learning rate for the optimizer
    epsilon: float = 0.1, # The epsilon for the epsilon-greedy policy
    gamma: float = 0.9, # The discount factor
    num_episodes: int = 100_000, # The number of episodes to train the agent
    hidden_size: int = 64, # The number of hidden units in the neural network
    batch_size: int = 32, # Number of possible experiences to sample from the replay buffer
    buffer_capacity: int = 10_000, # The capacity of the replay buffer
) -> QNetwork:
    num_states = len(env.S) # The number of states in the environment
    num_actions = len(env.A) # The number of actions in the environment

    # Un seul réseau pour estimer les Q-values ( => moving target pb)
    q_net = QNetwork(num_states, hidden_size, num_actions)
    
    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    replay_buffer = ReplayBuffer(buffer_capacity)

    for episode in range(num_episodes):
        s = env.reset()
                               
        while not env.is_game_over():
            a = choose_action_epsilon_greedy(
                s, env.available_actions(), q_net, num_states, epsilon
            )

            #take chosen action 
            env.step(a)
            #get the next state
            s_prime = env.state
            #get the reward
            r = env.inner_score
            #check if the episode is over
            done = env.is_game_over()
                     
            # Stockage de la transition dans le replay buffer
            replay_buffer.push(s, a, r, s_prime, done)
            s = s_prime

            # On entraîne uniquement si on a assez d'expériences dans le replay buffer
            if len(replay_buffer) < batch_size:
                continue

            # Mini-batch
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

            # Conversion en tenseurs PyTorch
            X      = torch.tensor(np.array([one_hot(st, num_states) for st in states]))       # batch_size (32) * number of possible  states 5
            X_next = torch.tensor(np.array([one_hot(st, num_states) for st in next_states]))  # batch_size (32) * number of possible  states 5
            actions_t = torch.tensor(actions, dtype=torch.long)    # batch_size (32)
            rewards_t = torch.tensor(rewards)                      # batch_size (32)
            dones_t   = torch.tensor(dones)                        # batch_size (32)

            # Q-values actuelles pour pour chaque action pour chaque transition dans le batch
            q_current = q_net(X)                 
            # 
            q_sa = q_current.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # batch_size (32)

            # TD target avec le MÊME réseau : r + γ * max Q(s', a')
            with torch.no_grad():
                q_next = q_net(X_next)                             # batch_size (32) * number of possible  actions 2
                max_q_next = q_next.max(dim=1).values              # batch_size (32)
                td_target = rewards_t + gamma * max_q_next * (1 - dones_t)

            # Gradient descendant (MSE loss)
            loss = loss_fn(q_sa, td_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return q_net
