from abc import ABC, abstractmethod

class BaseEnv(ABC):
    """Abstract base class for all environments."""
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
    def score(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_action_space(self) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def get_observation_space(self) -> list[int]:
        raise NotImplementedError

class ModelBasedEnv(BaseEnv, ABC):
    """Abstract base class for model-based environments."""
    def __init__(self, env_name):
        super().__init__(env_name)
        self.p = None 
        self.T = None

    @abstractmethod
    def num_states(self) -> int:
        pass

    @abstractmethod
    def num_actions(self) -> int:
        pass

    @abstractmethod
    def num_rewards(self) -> int:
        pass

    @abstractmethod
    def state_id(self, state) -> int:
        pass

class ModelFreeEnv(BaseEnv):
    """Abstract base class for model-free environments."""
    def __init__(self, env_name):
        super().__init__(env_name)