import torch
import os
import pygame
import numpy as np
from enum import IntEnum
from torch.nn import functional as F
import matplotlib.pyplot as plt
from deeprl_5iabd.config import settings


class Player(IntEnum):
    PLAYER_1 = 0
    PLAYER_2 = 1

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

def softmax_with_mask(S, M=None):
    if M is None:
        return F.softmax(S, dim=-1)
    M = M.detach() if isinstance(M, torch.Tensor) else torch.tensor(M).float()
    positive_or_null_s = S - S.min()
    masked_positive_or_null_s = positive_or_null_s * M
    negative_or_null_s = masked_positive_or_null_s - masked_positive_or_null_s.max()
    exp_s = torch.exp(negative_or_null_s)
    masked_exp_s = exp_s * M
    return masked_exp_s / masked_exp_s.sum()

def plot_metric(
    values,
    save_dir="results/plots",
    window_size=100,
    metric_name="metric",
    exp_name="exp",
    ylim=None,
    mask=None,
):
    os.makedirs(save_dir, exist_ok=True)

    values = np.asarray(values, dtype=float)
    n = len(values)
    valid = np.ones(n) if mask is None else np.asarray(mask, dtype=float)

    if window_size == 0:
        curve = np.where(valid > 0, values, np.nan)
        label = f"{metric_name} (raw)"
    else:
        w = max(1, min(window_size, n))

        def rolling_sum(a):
            c = np.cumsum(a).astype(float)
            out = c.copy()
            out[w:] -= c[:-w]
            return out

        num = rolling_sum(values * valid)
        den = rolling_sum(valid)
        curve = np.where(den > 0, num / np.maximum(den, 1), np.nan)
        label = f"{metric_name} (window={w})"

    plt.figure(figsize=(10, 5))
    plt.plot(curve, label=label)
    plt.title(f"{exp_name} - {metric_name}")
    plt.xlabel("Episodes")
    plt.ylabel(metric_name)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend()
    plt.grid(True, alpha=0.3)

    path = os.path.join(save_dir, f"{exp_name}_{metric_name}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_trace(trace, name="plot"):
    plt.figure()

    plt.plot(trace, marker='o')

    plt.title("Evolution du score")
    plt.xlabel("Step / Episode")
    plt.ylabel("Score")
    plt.yticks([-1, 0, 1])
    plt.grid()

    plt.savefig(f"{name}.png", bbox_inches='tight')

    plt.close()