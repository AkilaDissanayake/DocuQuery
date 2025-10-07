#Manage embeddings
import os
import numpy as np
from typing import List, Tuple   #For type hints
from openai import OpenAI        #OpenAI client to create embeddings

class EmbeddingsManager:
    def __init__(self, model_name: str = "text-embedding-3-small",api_key: str = None):
        self.model_name = model_name
        self.embeddings = None          #Hold all embeddings as a numpy array
        self.chunks_count = 0
        
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            raise RuntimeError("OPENAI_API_KEY not set in environment.")

    def create_embeddings(self, chunks: List[str], batch_size: int = 64):
        """Create embeddings for a list of chunk strings."""
        embs = []
        for i in range(0, len(chunks), batch_size):          #Process in batches to avoid API limits
            batch = chunks[i:i+batch_size]
            resp = self.client.embeddings.create(model=self.model_name, input=batch)
            embs.extend([d.embedding for d in resp.data])    #Extract embeddings
        self.embeddings = np.array(embs)   
        self.chunks_count = len(chunks)

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)    #Normalize vectors in a
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)    #Normalize vectors in b
        return np.dot(a_norm, b_norm.T)                                    #Dot product gives cosine similarity
    

    #Create embedding for the query
    def retrieve(self, query: str, k: int = 4) -> List[Tuple[int, float]]:
        """Return list of (chunk_index, score) for top-k similar chunks."""
        if self.embeddings is None:
            raise RuntimeError("Embeddings not created. Call create_embeddings first.")
        resp = self.client.embeddings.create(model=self.model_name, input=[query])
        q_emb = np.array(resp.data[0].embedding).reshape(1, -1)
        sims = self._cosine_sim(q_emb, self.embeddings)[0]     #Get similarity scores
        top_idx = np.argsort(-sims)[:k]                        #Get top-k indices
        return [(int(i), float(sims[i])) for i in top_idx]

    def save_npz(self, path: str):
        np.savez_compressed(path, embeddings=self.embeddings)

    def load_npz(self, path: str):
        data = np.load(path)
        self.embeddings = data["embeddings"]
        self.chunks_count = self.embeddings.shape[0]
