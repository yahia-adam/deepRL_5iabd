import torch
from mypythonlib.agents.base_agent import BaseAgent
from mypythonlib.helper import softmax_with_mask

class RandomPlayer(BaseAgent):
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def forward(self, x, mask):
        logits = torch.randn(self.action_dim)
        return softmax_with_mask(logits, mask)

    def save(self, filename):
        print("Random player is not a model, it is a random policy")
        
    @classmethod
    def load(cls, filename):
        print("Random player is not a model, it is a random policy")

if __name__ == "__main__":
    rp = RandomPlayer(action_dim=5)
    print(rp.forward(x=None, mask=[0,1,1,0,0]))
