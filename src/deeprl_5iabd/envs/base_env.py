from abc import ABC, abstractmethod

class BaseEnv(ABC):
    def __init__(self, env_name):
        self.env_name = env_name
        self.player = 0
    
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def is_game_over(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def score(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_action_space(self) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def get_observation_space(self) -> list[int]:
        raise NotImplementedError
