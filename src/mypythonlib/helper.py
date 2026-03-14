import torch
import torch.nn.functional as F

def get_default_device() -> str:
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        return torch.accelerator.current_accelerator().type
    return "cpu"

def softmax_with_mask(logits, mask):
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, device=logits.device).float()
    
    # S'assurer que mask et logits ont la même forme [batch, 32]
    mask = mask.view(logits.shape)

    # 1. Stabilité numérique : on soustrait le max des logits
    logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]

    # 2. Application du masque avec une valeur très petite
    # On utilise -1e9 plutôt que -inf pour éviter certains bugs de calcul
    masked_logits = logits.masked_fill(mask == 0, -1e9)

    # 3. Calcul du Softmax
    probs = F.softmax(masked_logits, dim=-1)

    # 4. Sécurité : si tout est à zéro à cause du masque, 
    # on force une distribution uniforme sur les actions autorisées
    if torch.isnan(probs).any() or probs.sum() <= 0:
        # On met 1 là où c'est autorisé, 0 ailleurs
        probs = mask / (mask.sum(dim=-1, keepdim=True) + 1e-9)

    return probs

class RandomPlayer():
    def __init__(self, a_len):
        self.a_len = a_len

    def play(self, mask):
        logits = torch.randn(self.a_len)
        return softmax_with_mask(logits, mask)

if __name__ == "__main__":
    rp = RandomPlayer(a_len=5)
    print(rp.play(mask=[0,1,1,0,0]))
