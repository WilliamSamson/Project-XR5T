from rich.progress import Progress
import time

import requests
import os
import json
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

CORE_NLP_URL = "http://localhost:5000/process_query"  # Core NLP endpoint

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
    console.print(GHOST_BANNER, style="bold green")


def simulate_typing_effect(text, style="orange", delay=0.05):
    for char in text:
        console.print(char, style=style, end="", flush=True)
        time.sleep(delay)
    console.print()  # Line break

def main():
    os.system("cls" if os.name == "nt" else "clear")
    display_banner()
    console.print(Panel("Welcome to the Ghost! Type 'exit' to quit.", title="[bold yellow]Greetings[/bold yellow]", style="bold magenta", expand=True), justify="center")

    while True:
        user_query = Prompt.ask("[bold magenta]Your query[/bold magenta]")
        if user_query.lower() == 'exit':
            simulate_typing_effect("Goodbye!", style="bold red")
            break

        with Progress(transient=True) as progress:
            task = progress.add_task("[bold cyan]Processing quhow to ery...[/bold cyan]", total=100)
            for i in range(100):
                time.sleep(0.02)  # Simulated progress
                progress.update(task, advance=1)

        try:
            response = requests.post(CORE_NLP_URL, json={"query": user_query})
            if response.status_code == 200:
                data = response.json()
                response_text = data.get("response", "No response available.")
                console.print(Panel(response_text, title="[bold green]Response[/bold green]", style="green", expand=True))
            else:
                console.print(f"[bold red]Error: {response.status_code}[/bold red]")
        except requests.exceptions.RequestException as e:
            console.print(f"[bold red]Connection error: {e}[/bold red]")

if __name__ == "__main__":
    main()
