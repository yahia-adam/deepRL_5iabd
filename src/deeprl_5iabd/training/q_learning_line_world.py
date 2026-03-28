import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from deeprl_5iabd.envs.line_world import LineWorld
from deeprl_5iabd.agents.q_learning import q_learning

console = Console()

def get_q_color(val: float, q_values: np.ndarray) -> str:
    diff = val - np.mean(q_values)
    intensity = min(abs(diff) / 5.0, 1.0)

    if diff >= 0:
        r = int(200 - intensity * 150)
        g = int(200 + intensity * 55)
        b = int(200 - intensity * 150)
    else:
        r = int(200 + intensity * 55)
        g = int(200 - intensity * 160)
        b = int(200 - intensity * 160)
    return f"#{r:02x}{g:02x}{b:02x}"

def render_lineworld_rich(Q, env):
    size = env.num_states()
    goal = size - 1
    trap = 0

    main_line = Table(
        show_header=False, 
        box=None, 
        padding=(0, 0),
        title="\n[bold] LINEWORLD : Valeurs Q par État[/bold]\n"
    )

    for _ in range(size):
        main_line.add_column(justify="center")

    cells = []
    for s in range(size):
        if s == trap:
            content = Text("\n💀\nTRAP", style="bold red", justify="center")
            cells.append(Panel(content, border_style="red", width=12))
        elif s == goal:
            content = Text("\n🏁\nGOAL", style="bold green", justify="center")
            cells.append(Panel(content, border_style="green", width=12))
        else:
            q_vals = Q[s]
            c_left = get_q_color(q_vals[0], q_vals)
            c_right = get_q_color(q_vals[1], q_vals)

            cell_text = Text()
            cell_text.append(f"← {q_vals[0]:+.2f}\n", style=f"bold {c_left}")
            cell_text.append(f"s{s}\n", style="dim")
            cell_text.append(f"→ {q_vals[1]:+.2f}", style=f"bold {c_right}")
            
            cells.append(Panel(cell_text, border_style="bright_black", width=12))

    main_line.add_row(*cells)

    console.print(main_line)
    console.print(
        "\n[dim]← Gauche (Action 0)  |  → Droite (Action 1)  |  "
        "[green]Vert[/green] = Meilleur choix[/dim]\n"
    )

if __name__ == "__main__":
    env = LineWorld()
    Q = q_learning(env=env)
    render_lineworld_rich(Q, env)