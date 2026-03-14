from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def forward(self, x, mask):
        raise NotImplementedError

    @abstractmethod
    def save(self, filename):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, filename):
        raise NotImplementedError
