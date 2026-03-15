import torch
import pygame
import torch.nn.functional as F
import numpy as np

class ImageButton:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.image = None

    def draw(self, screen):
        if self.image:
            screen.blit(self.image, self.rect)
        else:
            pygame.draw.rect(screen, (200, 200, 200), self.rect)

    def is_clicked(self, event):
        return event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos)

def get_default_device() -> str:
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        return torch.accelerator.current_accelerator().type
    return "cpu"

def softmax_with_mask(S, M):
    M = torch.tensor(M, dtype=S.dtype, device=S.device)
    positive_or_null_s = S - S.min()
    masked_positive_or_null_s = positive_or_null_s * M
    negative_or_null_s = masked_positive_or_null_s - masked_positive_or_null_s.max()
    exp_s = torch.exp(negative_or_null_s)
    masked_exp_s = exp_s * M
    return masked_exp_s / masked_exp_s.sum()