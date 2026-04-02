import torch
import pygame
from deeprl_5iabd.config import settings

class ImageButton:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.image = None
        self.score_text = None
        self.score_color = (200, 200, 200)

    def draw(self, screen):
        if self.image:
            screen.blit(self.image, self.rect)
        else:
            pygame.draw.rect(screen, (200, 200, 200), self.rect)
        if self.score_text:
            font = pygame.font.SysFont("Arial", 50, bold=True)
            text_surface = font.render(self.score_text, True, self.score_color)
            text_rect = text_surface.get_rect(center=self.rect.center)
            screen.blit(text_surface, text_rect)

    def is_clicked(self, event):
        return event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos)

def softmax_with_mask(S, M):
    positive_or_null_s = S - S.min()
    masked_positive_or_null_s = positive_or_null_s * M
    negative_or_null_s = masked_positive_or_null_s - masked_positive_or_null_s.max()
    exp_s = torch.exp(negative_or_null_s)
    masked_exp_s = exp_s * M
    return masked_exp_s / masked_exp_s.sum()
