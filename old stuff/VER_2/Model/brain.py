import os
import json
import hashlib
import ast
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from rich.progress import Progress
from rich.console import Console
from rich.panel import Panel
from sentence_transformers import SentenceTransformer  # pip install sentence-transformers


# -------------------------
# Core Memory System
# -------------------------

class VectorMemory:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.memory = {}
        self.vectors = np.empty((0, 384))

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def store(self, text: str, metadata: Dict) -> str:
        vector = self.encoder.encode(text)
        mem_id = self._hash(text)
        self.memory[mem_id] = {
            'text': text,
            'vector': vector,
            'timestamp': datetime.now().isoformat(),
            **metadata
        }
        self.vectors = np.vstack([self.vectors, vector])
        return mem_id

    def recall(self, query: str, top_k=3) -> List[Dict]:
        query_vec = self.encoder.encode(query)
        similarities = np.dot(self.vectors, query_vec)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.memory[list(self.memory.keys())[idx]] for idx in top_indices]


# -------------------------
# Personality Simulation
# -------------------------

class CognitiveModel:
    def __init__(self, profile_path: str):
        self.decision_patterns = self._load_profile(profile_path)
        self.reasoning_steps = []

    def _load_profile(self, path: str) -> Dict:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return {'decision_weights': {}, 'reasoning_templates': []}

    def analyze(self, query: str, context: List[Dict]) -> Dict:
        self.reasoning_steps = []
        # Simple pattern matching - expand with actual ML model
        best_match = max(context, key=lambda x: len(x['text']), default={'text': ''})
        return {
            'raw_query': query,
            'context': best_match,
            'reasoning': self._apply_thinking_patterns(query),
            'decision': self._make_decision(query)
        }

    def _apply_thinking_patterns(self, query: str) -> List[str]:
        # Expand with actual cognitive modeling
        return [
            f"Pattern matching: {query[:5]}...",
            "Cross-referencing with historical decisions",
            "Generating hypothetical scenarios"
        ]


# -------------------------
# Self-Modification System
# -------------------------

class CodeGenerator:
    def __init__(self):
        self.sandbox = self._create_sandbox()
        self.allowed_actions = ['file_io', 'math', 'data_processing']

    def _create_sandbox(self):
        # Restricted execution environment
        return {
            '__builtins__': {
                'print': print,
                'range': range,
                'len': len,
                'str': str,
                'int': int,
                'float': float
            }
        }

    def generate(self, task: str) -> Optional[str]:
        try:
            parsed = ast.parse(task)
            # Basic safety check
            if any(isinstance(node, ast.Import) for node in parsed.body):
                return None
            return compile(parsed, filename="<generated>", mode="exec")
        except:
            return None


# -------------------------
# Integrated System
# -------------------------

class XR5T:
    def __init__(self, profile_path: str = "personality.json"):
        self.memory = VectorMemory()
        self.personality = CognitiveModel(profile_path)
        self.coder = CodeGenerator()
        self.console = Console()

        # Initial priming
        self._prime_memory()

    def _prime_memory(self):
        base_knowledge = [
            ("Problem solving approach: Divide and conquer", {"type": "strategy"}),
            ("Preferred communication style: Direct and analytical", {"type": "style"}),
            ("Common decision patterns: Risk-averse in personal matters", {"type": "pattern"})
        ]
        for text, meta in base_knowledge:
            self.memory.store(text, meta)

    def process(self, query: str) -> Dict:
        context = self.memory.recall(query)
        analysis = self.personality.analyze(query, context)

        if "#generate" in query:
            analysis['code'] = self._handle_code_generation(query)

        self._update_memory(analysis)
        return analysis

    def _handle_code_generation(self, query: str) -> Dict:
        task = query.split("#generate")[1].strip()
        code_obj = self.coder.generate(task)
        if not code_obj:
            return {"error": "Invalid code generation request"}

        try:
            exec(code_obj, self.coder.sandbox)
            return {"success": True, "output": "Execution complete"}
        except Exception as e:
            return {"error": str(e)}


# -------------------------
# Enhanced UI Components
# -------------------------

class GhostInterface:
    def __init__(self):
        self.console = Console()
        self.xr5t = XR5T()

    def _format_response(self, analysis: Dict) -> str:
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