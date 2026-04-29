import sys
import subprocess
import pygame

pygame.init()
screen = pygame.display.set_mode((480, 380))
pygame.display.set_caption("DeepRL")
font  = pygame.font.SysFont(None, 26)
clock = pygame.time.Clock()

BUTTONS = [
    ("Line World  -  Humain",               "lineworld",  "human"),
    ("Line World  -  Aleatoire",            "lineworld",  "random"),
    ("Grid World  -  Humain",               "gridworld",  "human"),
    ("Grid World  -  Aleatoire",            "gridworld",  "random"),
    ("Tic-Tac-Toe  -  Humain vs Humain",    "tictactoe",  "hvh"),
    ("Tic-Tac-Toe  -  Humain vs Aleatoire", "tictactoe",  "hvr"),
    ("Quarto  -  Humain vs Humain",         "quarto",     "hvh"),
    ("Quarto  -  Humain vs Aleatoire",      "quarto",     "hvr"),
    ("Quarto  -  Aleatoire vs Aleatoire",   "quarto",     "rvr"),
]

def launch(env, mode):
    if env == "lineworld":
        from deeprl_5iabd.envs.line_world import LineWorldEnv, human_player, random_player
        e = LineWorldEnv(render_mode="human")
        human_player(e) if mode == "human" else random_player(e)
        e.close()

    elif env == "gridworld":
        from deeprl_5iabd.envs.grid_world import GridWorldEnv, human_player, random_player
        e = GridWorldEnv(render_mode="human")
        human_player(e) if mode == "human" else random_player(e)
        e.close()

    elif env == "tictactoe":
        from deeprl_5iabd.envs.tictactoe import TicTacToeEnv, human_vs_human, human_vs_random
        e = TicTacToeEnv(render_mode="human")
        human_vs_human(e) if mode == "hvh" else human_vs_random(e)
        e.close()

    elif env == "quarto":
        from deeprl_5iabd.envs.quarto import QuartoEnv, human_vs_human, human_vs_random, random_vs_random
        e = QuartoEnv(render_mode="human")
        if mode == "hvh":   human_vs_human(e)
        elif mode == "hvr": human_vs_random(e)
        else:               random_vs_random(e)
        e.close()

BTN_W, BTN_H = 420, 30
BTN_X  = (480 - BTN_W) // 2
START_Y = 60
GAP    = 4

while True:
    mx, my = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); sys.exit()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            pygame.quit(); sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i, (label, env, mode) in enumerate(BUTTONS):
                rect = pygame.Rect(BTN_X, START_Y + i * (BTN_H + GAP), BTN_W, BTN_H)
                if rect.collidepoint(mx, my):
                    pygame.quit()
                    launch(env, mode)
                    pygame.init()
                    screen = pygame.display.set_mode((480, 380))
                    pygame.display.set_caption("DeepRL")
                    font  = pygame.font.SysFont(None, 26)
                    clock = pygame.time.Clock()

    screen.fill((0, 0, 0))

    title = font.render("DeepRL - Launcher", True, (255, 255, 255))
    screen.blit(title, (480 // 2 - title.get_width() // 2, 20))

    for i, (label, env, mode) in enumerate(BUTTONS):
        rect = pygame.Rect(BTN_X, START_Y + i * (BTN_H + GAP), BTN_W, BTN_H)
        hovered = rect.collidepoint(mx, my)
        pygame.draw.rect(screen, (60, 60, 60) if hovered else (30, 30, 30), rect)
        pygame.draw.rect(screen, (80, 80, 80), rect, 1)
        txt = font.render(label, True, (255, 255, 255) if hovered else (180, 180, 180))
        screen.blit(txt, (rect.x + 10, rect.centery - txt.get_height() // 2))

    pygame.display.flip()
    clock.tick(60)