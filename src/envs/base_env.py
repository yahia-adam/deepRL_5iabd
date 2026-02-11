from abc import ABC, abstractmethod

class BaseEnv(ABC):
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = None

    @abstractmethod
    def step(self, data):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_action_space(self):
        pass

    @abstractmethod
    def get_observation_space(self):
        pass

    @abstractmethod
    def monitor(self, is_monitor, is_train, video_record_dir="", record_video_every=10):
        pass

    @abstractmethod
    def render(self):
        pass
