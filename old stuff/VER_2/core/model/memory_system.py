import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
import hashlib
from datetime import datetime


class VectorMemory:
    def __init__(self, persist_dir="data/memory_cache"):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.persist_dir = persist_dir
        self.memory = {}
        self.vectors = np.empty((0, 384))
        self._load_memory()

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def _load_memory(self):
        if os.path.exists(f"{self.persist_dir}/memory.pkl"):
            with open(f"{self.persist_dir}/memory.pkl", "rb") as f:
                data = pickle.load(f)
                self.memory = data['memory']
                self.vectors = data['vectors']

    def store(self, text: str, metadata: dict) -> str:
        vector = self.encoder.encode(text)
        mem_id = self._hash(text)
        self.memory[mem_id] = {
            'text': text,
            'vector': vector,
            'timestamp': datetime.now().isoformat(),
            **metadata
        }
        self.vectors = np.vstack([self.vectors, vector])
        self._save_memory()
        return mem_id

    def _save_memory(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        with open(f"{self.persist_dir}/memory.pkl", "wb") as f:
            pickle.dump({'memory': self.memory, 'vectors': self.vectors}, f)

    def recall(self, query: str, top_k=3) -> list:
        """Retrieve relevant memories using vector similarity"""
        if not self.memory or len(self.memory) == 0:
            return []

        # Ensure we don't request more items than available
        actual_k = min(top_k, len(self.memory))

        query_vec = self.encoder.encode(query)
        similarities = np.dot(self.vectors, query_vec)

        # Get indices of top_k most similar vectors
        top_indices = np.argsort(similarities)[-actual_k:][::-1]

        # Safely retrieve memories using valid indices
        memory_keys = list(self.memory.keys())
        return [self.memory[memory_keys[idx]] for idx in top_indices
                if idx < len(memory_keys)]