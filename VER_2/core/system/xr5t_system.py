# core/system/xr5t_system.py
from core.model.memory_system import VectorMemory
from core.model.cognitive_model import CognitiveModel
from core.model.code_generator import CodeGenerator
import json


class XR5T:
    def __init__(self, profile_path: str = "data/personality.json"):
        self.memory = VectorMemory()
        self.personality = CognitiveModel(profile_path)
        self.coder = CodeGenerator()
        self._prime_memory()

    def _prime_memory(self):
        base_knowledge = [
            ("Coding pattern: Test-driven development", {"type": "methodology"}),
            ("Debugging approach: Rubber duck debugging", {"type": "strategy"}),
            ("Documentation style: API-first design", {"type": "preference"})
        ]
        for text, meta in base_knowledge:
            self.memory.store(text, meta)

    def process(self, query: str) -> dict:
        context = self.memory.recall(query)
        analysis = self.personality.analyze(query, context)

        if query.lower().startswith("add pattern:"):
            return self._handle_new_pattern(query)
        elif query.lower().startswith("my name is"):
            return self._store_personal_data(query)

        if query.startswith("#generate"):
            code_task = query.replace("#generate", "").strip()
            analysis['code'] = self.coder.generate(code_task)

        self._update_memory(analysis)
        return analysis

    def _update_memory(self, analysis: dict):
        for insight in analysis.get('learning_points', []):
            self.memory.store(insight, {"type": "insight"})

    def _handle_new_pattern(self, query: str) -> dict:
        pattern = query.split(":", 1)[1].strip()
        self.memory.store(f"User-defined pattern: {pattern}",
                          {"type": "personal_pattern"})
        return {
            "reasoning": ["Pattern storage initiated"],
            "decision": f"New pattern '{pattern}' added to memory"
        }

    def _store_personal_data(self, query: str) -> dict:
        name = query.replace("my name is", "").strip()
        self.memory.store(f"User identity: {name}",
                          {"type": "personal_info", "priority": 10})
        return {
            "reasoning": ["Personal data storage protocol activated"],
            "decision": f"Identity set to: {name}"
        }