import torch
import torch.nn.functional as F

class RandomPlayer():
    def __init__(self, a_len):
        self.a_len = a_len
    
    def play(self, mask):
        logits = torch.randn(self.a_len)
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        logits.masked_fill_(~mask_tensor, float('-inf'))
        probs = F.softmax(logits, dim=0)
        print(mask)
        print(probs)
        return probs

if __name__ == "__main__":
    rp = RandomPlayer(5)
    print(rp.play([0,1,1,0,0]))
