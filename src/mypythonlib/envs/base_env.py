from abc import ABC, abstractmethod

class BaseEnv(ABC):
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = None
        self.current_player = 0
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def step(self, action: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def is_game_over(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def score(self):
        raise NotImplementedError

    @abstractmethod
    def get_action_space(self):
        raise NotImplementedError

    @abstractmethod
    def get_observation_space(self):
        raise NotImplementedError

    @abstractmethod
    def monitor(self, is_train, win_rate, episode_length, policy_loss):
        raise NotImplementedError

    @abstractmethod
    def render(self):
        raise NotImplementedError
