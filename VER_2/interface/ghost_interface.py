import sys
import os
import time
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from core.system.xr5t_system import XR5T


class GhostInterface:
    def __init__(self):
        self.console = Console()
        self.xr5t = XR5T()

    def _format_response(self, analysis: dict) -> str:
        reasoning = "\n".join(analysis.get('reasoning', []))
        return f"""\n[Analysis]
{reasoning}

[Decision]
{analysis.get('decision', 'No conclusion reached')}"""

    def run(self):
        self.console.print(Panel("Ghost Interface - XR5T Core Online", style="bold cyan"))
        while True:
            try:
                query = self.console.input("[bold magenta]>>[/bold magenta] ")
                if query.lower() in ['exit', 'quit']:
                    break

                with Progress(transient=True) as progress:
                    task = progress.add_task("[cyan]Processing...", total=100)
                    for _ in range(100):
                        time.sleep(0.01)
                        progress.update(task, advance=1)

                result = self.xr5t.process(query)
                response = self._format_response(result)
                self.console.print(Panel(response, title="[bold green]Analysis[/bold green]", expand=False))

            except KeyboardInterrupt:
                self.console.print("\n[bold red]Shutdown initiated...[/bold red]")
                break


if __name__ == "__main__":
    GhostInterface().run()