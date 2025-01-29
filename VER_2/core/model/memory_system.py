import numpy as np
from sentence_transformers import SentenceTransformer
import hashlib
from datetime import datetime


class VectorMemory:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.memory = {}
        self.vectors = np.empty((0, 384))

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

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
        return mem_id

    def recall(self, query: str, top_k=3) -> list:
        query_vec = self.encoder.encode(query)
        similarities = np.dot(self.vectors, query_vec)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.memory[list(self.memory.keys())[idx]] for idx in top_indices]