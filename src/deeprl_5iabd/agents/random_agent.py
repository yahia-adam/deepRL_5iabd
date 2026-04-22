from gymnasium.spaces import Discrete

class RandomPlayer(Discrete):
    def __init__(self, n):
        super().__init__(n)
