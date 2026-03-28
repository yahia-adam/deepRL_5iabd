import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from deeprl_5iabd.envs.tictactoe import TicTacToe
from deeprl_5iabd.agents.q_learning import q_learning

console = Console()

def render_q_analysis(Q, env, board_flat, title="Analyse d'état"):
    s_id = env.state_id(board_flat)
    q_values = Q[s_id]
    
    table = Table(show_header=False, box=None, title=f"\n[bold cyan]🔍 {title}[/bold cyan]")
    for _ in range(3): table.add_column(justify="center", width=20)

    grid = board_flat.reshape(3, 3)
    for r in range(3):
        row_cells = []
        for c in range(3):
            idx = r * 3 + c
            val = grid[r, c]
            q_val = q_values[idx]

            if val == 0:
                content = Text("\n⭕\n(Agent - P0)", style="bold green", justify="center")
                style = "green"
            elif val == 1:
                content = Text("\n❌\n(Random - P1)", style="bold red", justify="center")
                style = "red"
            else:
                is_best = (q_val == max(q_values[board_flat == -1])) if any(board_flat == -1) else False
                color = "bright_green" if is_best else "white"
                content = Text(f"\nQ: {q_val:.3f}\n(Vide)", style=f"bold {color}", justify="center")
                style = "bright_black"

            row_cells.append(Panel(content, title=f"Case {idx}", border_style=style, height=7))
        table.add_row(*row_cells)
    console.print(table)

if __name__ == "__main__":
    env = TicTacToe()
    
    console.print("[bold yellow] Entraînement du Q-Learning (Agent ⭕ contre Random ❌)...[/bold yellow]")
    Q = q_learning(env=env, num_episodes=100_000, epsilon=0.1, learning_rate=0.1, is_two_players=True)
    console.print("[bold green] Entraînement terminé ![/bold green]")

    tests = [
        (np.array([1, -1, 1, 0, 0, -1, -1, -1, -1]), "Finir ligne milieu"),
        (np.array([1, 1, -1, 0, -1, -1, -1, -1, -1]), "Bloquer ligne haut"),
        (np.array([1, -1, -1, -1, 0, -1, -1, -1, 1]), "Contrer fourchette coins"),
    ]

    for board, desc in tests:
        render_q_analysis(Q, env, board, desc)