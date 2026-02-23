import torch
import torch.nn.functional as F

def softmax_with_mask(x, mask):
    mask_tensor = torch.tensor(mask, dtype=torch.bool)
    x.masked_fill_(~mask_tensor, float('-inf'))
    return F.softmax(x, dim=0)

class RandomPlayer():
    def __init__(self, a_len):
        self.a_len = a_len
    
    def play(self, mask):
        logits = torch.randn(self.a_len)
        return softmax_with_mask(logits, mask)
    
if __name__ == "__main__":
    rp = RandomPlayer(5)
    print(rp.play([0,1,1,0,0]))
