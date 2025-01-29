import numpy as np
from sentence_transformers import SentenceTransformer
import hashlib
from datetime import datetime
import os


class VectorMemory:
    def __init__(self, persist_dir: str = "data/memory_cache"):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.persist_dir = persist_dir
        self.memory = {}
        self.vectors = np.empty((0, 384))  # Initialize vectors array
        self._load_persistent_memory()

    def _load_persistent_memory(self):
        if not os.path.exists(self.persist_dir):
            os.makedirs(self.persist_dir)
        # Implement actual loading logic here

    def store(self, text: str, metadata: dict) -> str:
        """Store new memory with vector embedding"""
        vector = self.encoder.encode(text)
        mem_id = hashlib.sha256(text.encode()).hexdigest()
        self.memory[mem_id] = {
            'text': text,
            'vector': vector,
            'timestamp': datetime.now().isoformat(),
            **metadata
        }
        self.vectors = np.vstack([self.vectors, vector])
        return mem_id

    def recall(self, query: str, top_k=3) -> list:
        """Retrieve relevant memories using vector similarity"""
        query_vec = self.encoder.encode(query)
        similarities = np.dot(self.vectors, query_vec)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.memory[list(self.memory.keys())[idx]] for idx in top_indices]