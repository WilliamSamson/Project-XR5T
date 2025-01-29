from core.model.memory_system import VectorMemory
from core.model.cognitive_model import CognitiveModel
import json
import os


class XR5T:
    def __init__(self, profile_path: str = "/home/kayode-olalere/PycharmProjects/Project XR5T/VER_2/data/personality.json"):
        self.memory = VectorMemory()
        self.personality = CognitiveModel(profile_path)
        self._prime_memory()

    def _prime_memory(self):
        base_knowledge = [
            ("Problem solving approach: Divide and conquer", {"type": "strategy"}),
            ("Preferred communication style: Direct and analytical", {"type": "style"}),
            ("Common decision patterns: Risk-averse in personal matters", {"type": "pattern"})
        ]
        for text, meta in base_knowledge:
            self.memory.store(text, meta)

    def process(self, query: str) -> dict:
        context = self.memory.recall(query)
        return self.personality.analyze(query, context)