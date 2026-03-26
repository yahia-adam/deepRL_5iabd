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
    """Affiche une grille 3x3 avec les valeurs Q pour chaque case."""
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
                content = Text("\n❌\n(Joueur 0)", style="bold blue", justify="center")
                style = "blue"
            elif val == 1:
                content = Text("\n⭕\n(Joueur 1)", style="bold red", justify="center")
                style = "red"
            else:
                # Couleur selon la qualité du coup
                color = "green" if q_val == max(q_values) else "white"
                content = Text(f"\nQ: {q_val:.3f}\n(Vide)", style=f"bold {color}", justify="center")
                style = "bright_black"

            row_cells.append(Panel(content, title=f"Case {idx}", border_style=style, height=7))
        table.add_row(*row_cells)
    console.print(table)

if __name__ == "__main__":
    # 1. Initialisation de l'environnement
    env = TicTacToe()
    
    # 2. Entraînement
    console.print("[bold yellow]🚀 Entraînement du Q-Learning (100k épisodes)...[/bold yellow]")
    # Note: Assure-toi que ton q_learning retourne bien la Q-Table
    Q = q_learning(env=env, num_episodes=10, epsilon=0.2, learning_rate=0.1)
    console.print("[bold green]✅ Entraînement terminé ![/bold green]")

    # 3. Tests de "cerveau" de l'IA
    # Situation A : L'IA (P0) doit finir la ligne du haut
    win_move = np.array([0, 0, -1, 1, 1, -1, -1, -1, -1])
    render_q_analysis(Q, env, win_move, "Test : Victoire immédiate (doit choisir Case 2)")

    # Situation B : L'IA (P0) doit bloquer P1 en bas
    block_move = np.array([0, -1, -1, -1, 0, -1, 1, 1, -1])
    render_q_analysis(Q, env, block_move, "Test : Blocage défensif (doit choisir Case 8)")

    # 4. Lancement du jeu interactif
    console.print("\n[bold magenta]🎮 Lancement de l'interface Pygame...[/bold magenta]")
    
    # Pour jouer contre ton agent Q-Learning, 
    # tu peux modifier play_vs_random dans TicTacToe pour utiliser ta Q-Table
    env.play_vs_random()