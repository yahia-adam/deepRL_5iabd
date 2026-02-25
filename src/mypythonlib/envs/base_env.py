from abc import ABC, abstractmethod

class BaseEnv(ABC):
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = None
        self.current_player = 0
    
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, data):
        pass

    @abstractmethod
    def is_game_over(self, data):
        pass

    @abstractmethod
    def score(self):
        pass

    @abstractmethod
    def get_action_space(self):
        pass

    @abstractmethod
    def get_observation_space(self):
        pass

    @abstractmethod
    def monitor(self, is_train, win_rate, episode_length, policy_loss):
        pass

    @abstractmethod
    def render(self):
        pass
