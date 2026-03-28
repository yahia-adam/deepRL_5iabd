import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from deeprl_5iabd.envs.grid_world import GridWorld
from deeprl_5iabd.agents.q_learning import q_learning

console = Console()

def get_q_color(val: float, q_values: np.ndarray) -> str:
    avg = np.mean(q_values)
    diff = val - avg
    intensity = min(abs(diff) * 2, 1.0) 

    if diff >= 0:
        r = int(200 - intensity * 150)
        g = int(200 + intensity * 55)
        b = int(200 - intensity * 150)
    else:
        r = int(200 + intensity * 55)
        g = int(200 - intensity * 160)
        b = int(200 - intensity * 160)
    return f"#{r:02x}{g:02x}{b:02x}"

def render_gridworld_rich(Q, env):
    size = 5
    goal = (0, 4)
    trap = (4, 4)
    arrows = {0: "▼", 1: "▲", 2: "▶", 3: "◀"}

    main_grid = Table(
        show_header=False, 
        box=None, 
        padding=0, 
        collapse_padding=True,
        title="\n[bold] GRIDWORLD : Politique & Valeurs Q[/bold]\n"
    )
    
    for _ in range(size):
        main_grid.add_column(justify="center")

    for r in range(size):
        row_cells = []
        for c in range(size):
            state_idx = r * size + c
            
            if (r, c) == goal:
                cell_content = Panel(Text("\n🏁\nGOAL", style="bold green", justify="center"), border_style="green", height=7)
            elif (r, c) == trap:
                cell_content = Panel(Text("\n💀\nTRAP", style="bold red", justify="center"), border_style="red", height=7)
            else:
                q_vals = Q[state_idx]
                best_action = np.argmax(q_vals)
                
                mini_table = Table.grid(expand=True)
                mini_table.add_column(width=7, justify="center")
                mini_table.add_column(width=7, justify="center")
                mini_table.add_column(width=7, justify="center")

                colors = [get_q_color(v, q_vals) for v in q_vals]

                mini_table.add_row("", Text(f"{q_vals[1]:.2f}", style=f"bold {colors[1]}"), "")
                mini_table.add_row(
                    Text(f"{q_vals[3]:.2f}", style=f"bold {colors[3]}"), 
                    Text(arrows[best_action], style="bold yellow underline"), 
                    Text(f"{q_vals[2]:.2f}", style=f"bold {colors[2]}")
                )
                mini_table.add_row("", Text(f"{q_vals[0]:.2f}", style=f"bold {colors[0]}"), "")
                
                cell_content = Panel(mini_table, border_style="bright_black", height=7)
            
            row_cells.append(cell_content)
        main_grid.add_row(*row_cells)

    console.print(main_grid)
    console.print("\n[dim]Légende: ▲▼◀▶ = Meilleure action | Chiffres = Valeurs Q par direction[/dim]\n")

if __name__ == "__main__":
    env = GridWorld()
    Q = q_learning(env=env, num_episodes=500_000)
    render_gridworld_rich(Q, env)