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

def ema(values, alpha=0.1):
    """Exponential moving average (lissage type TensorBoard)"""
    smoothed = []
    v = values[0]
    for x in values:
        v = alpha * x + (1 - alpha) * v
        smoothed.append(v)
    return np.array(smoothed)


def plot_rl_dashboard(
    reward_per_episode,
    loss_per_episode,
    algo_name="algo",
    env_name="env",
    prams = None,
    window=100,
    reward_threshold=0,
    ema_alpha=0.1,
):
    n = len(reward_per_episode)

    avg_reward, success, failure, avg_loss, variance = [], [], [], [], []

    for i in range(0, n, window):
        r = np.array(reward_per_episode[i:i+window])
        l = np.array(loss_per_episode[i:i+window])

        if len(r) == 0:
            continue

        avg_reward.append(np.mean(r))
        success.append(np.sum(r > reward_threshold))
        failure.append(np.sum(r < reward_threshold))
        avg_loss.append(np.mean(l))
        variance.append(np.var(r))

    x = np.arange(len(avg_reward)) * window

    # =========================
    # Lissage EMA (style TensorBoard)
    # =========================
    avg_reward_s = ema(avg_reward, ema_alpha)
    avg_loss_s = ema(avg_loss, ema_alpha)
    success_s = ema(success, ema_alpha)
    failure_s = ema(failure, ema_alpha)
    variance_s = ema(variance, ema_alpha)

    # =========================
    # FIGURE DASHBOARD
    # =========================
    fig = plt.figure(figsize=(14, 10))

    # ---- Reward ----
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(x, avg_reward_s, label="Reward moyen (EMA)")
    ax1.set_title("Reward moyen")
    ax1.set_xlabel("Episodes")
    ax1.legend()

    # ---- Success / Failure ----
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(x, success_s, label="Succès / 100 épisodes (EMA)")
    ax2.plot(x, failure_s, label="Échecs / 100 épisodes (EMA)")
    ax2.set_title("Performance (comptage)")
    ax2.set_xlabel("Episodes")
    ax2.legend()

    # ---- Loss ----
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(x, avg_loss_s, label="Loss moyenne (EMA)")
    ax3.set_title("Loss de la politique")
    ax3.set_xlabel("Episodes")
    ax3.legend()

    # ---- Variance ----
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(x, variance_s, label="Variance des rewards (EMA)")
    ax4.set_title("Stabilité de l’apprentissage")
    ax4.set_xlabel("Episodes")
    ax4.legend()

    plt.suptitle(f"{algo_name} {env_name} {prams}", fontsize=16)
    plt.tight_layout()

    # =========================
    # SAVE IMAGE
    # =========================
    save_path = f"{settings.training_logs_dir}/{algo_name}/{env_name}"
    os.makedirs(save_path, exist_ok=True)
    file = os.path.join(save_path, f"{algo_name} {env_name} {prams}.png")
    plt.savefig(file, dpi=300)
    print(f"Dashboard sauvegardé : {file}")
