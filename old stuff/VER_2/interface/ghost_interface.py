# interface/ghost_interface.py
import sys
import os
import time
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.syntax import Syntax

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from core.system.xr5t_system import XR5T


console = Console()
title = "X-R-5-T"

GHOST_BANNER = """
  ██████╗   ██╗  ██╗    ██████╗     ████████╗  ████████╗
 ██╔════╝   ██║  ██║  ██╔═══  ██╗  ██╔════╝    ╚══██╔══╝
 ██║  ███╗  ███████║  ██║     ██║  ███████╗       ██║   
 ██║   ██║  ██╔══██║  ██║     ██║     ╚═══██╗     ██║   
 ╚██████╔╝  ██║  ██║   ╚██████╔╝   ████████╔╝     ██║   
  ╚═════╝   ╚═╝  ╚═╝    ╚═════╝    ╚═════╝        ╚═╝   
"""

def display_banner():
    console.print(GHOST_BANNER, style="bold cyan")


class GhostInterface:
    def __init__(self):

        self.console = Console()
        self.xr5t = XR5T()
      #  os.system("cls" if os.name == "nt" else "clear")
        #display_banner()
        self.session_history = []

    def _format_code(self, code: str) -> Syntax:
        return Syntax(code, "python", theme="monokai", line_numbers=True)

    def run(self):
        self.console.print(Panel("XR5T v2.0 - Cognitive Interface Active", style="bold blue"))
        while True:
            try:
                query = self.console.input("\n[bold cyan]Query[/bold cyan] > ")
                if query.lower() in ('exit', 'quit'):
                    break

                # Process query
                with Progress(transient=True) as progress:
                    task = progress.add_task("[magenta]Analyzing...", total=100)
                    for _ in range(100):
                        time.sleep(0.01)
                        progress.update(task, advance=1)

                result = self.xr5t.process(query)
                self._display_result(result)

            except KeyboardInterrupt:
                self.console.print("\n[red]Security Protocol: Session Terminated[/red]")
                break

    def _display_result(self, result: dict):
        # Display reasoning
        self.console.print(Panel(
            "\n".join(result['reasoning']),
            title="[bold green]Cognitive Process[/bold green]",
            border_style="bright_yellow"
        ))

        # Display decision
        self.console.print(Panel(
            result['decision'],
            title="[bold blue]Recommended Action[/bold blue]",
            border_style="cyan"
        ))

        # Display code if generated
        if 'code' in result:
            if 'error' in result['code']:
                self.console.print(f"[red]Code Error: {result['code']['error']}[/red]")
            else:
                self.console.print(Panel(
                    self._format_code(result['code']['code']),
                    title="[bold magenta]Generated Code[/bold magenta]",
                    border_style="bright_green"
                ))


if __name__ == "__main__":
    GhostInterface().run()