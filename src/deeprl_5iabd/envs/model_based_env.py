from abc import ABC, abstractmethod
from deeprl_5iabd.envs.base_env import BaseEnv

class ModelBasedEnv(BaseEnv, ABC):
    def __init__(self, env_name):
        super().__init__(env_name)
        self.p_matrix = None 
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
