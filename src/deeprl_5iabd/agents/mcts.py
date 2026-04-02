class Node:
    def __init__(self, action=None):
        self.action = action
        self.visits = 0
        self.value = 0
        self.children = []

def monte_carlo_tree_search(env: BaseEnv, random_agent: RandomPlayer, num_rollouts: int):
    pass