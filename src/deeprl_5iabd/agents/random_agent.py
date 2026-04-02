import torch
from deeprl_5iabd.agents.base_agent import BaseAgent
from deeprl_5iabd.helper import softmax_with_mask
from deeprl_5iabd.config import settings

class RandomPlayer(BaseAgent):
    def __init__(self, action_dim):
        self.name = "RandomPlayer"
        self.action_dim = action_dim

    def forward(self, x, mask):
        logits = torch.randn(self.action_dim, device=settings.device)
        return softmax_with_mask(logits, mask)

    def save(self, filename):
        print("Random player is not a model, it is a random policy")

    def clone(self, name=None):
        return RandomPlayer(action_dim=self.action_dim)

    @classmethod
    def load(cls, filename):
        print("Random player is not a model, it is a random policy")

if __name__ == "__main__":
    rp = RandomPlayer(action_dim=5)
    print(rp.forward(x=None, mask=[0,1,1,0,0]))
