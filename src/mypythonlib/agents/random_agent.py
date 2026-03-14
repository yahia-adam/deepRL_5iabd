import torch
from mypythonlib.agents.agent_base import MyModel
from mypythonlib.helper import softmax_with_mask

class RandomPlayer(MyModel):
    def __init__(self, a_len):
        self.a_len = a_len
        super().__init__(input_size=a_len, output_size=a_len, hiddenlayers=[1,1])

    def forward(self, x, mask):
        logits = torch.randn(self.a_len)
        return softmax_with_mask(logits, mask)

    def save(self, filename="random_player.pth"):
        print("random player is not a model, it is a random policy")
        raise NotImplementedError

    @classmethod
    def load(cls, filename="random_player.pth"):
        print("random player is not a model, it is a random policy")
        raise NotImplementedError

if __name__ == "__main__":
    rp = RandomPlayer(a_len=5)
    print(rp.forward(x=None, mask=[0,1,1,0,0]))
