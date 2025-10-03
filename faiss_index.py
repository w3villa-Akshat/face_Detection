import faiss
import numpy as np
from db import get_all_embeddings, add_embedding


class FaissIndexManager:
    def __init__(self, index_path="faiss.index"):
        self.index_path = index_path
        self.index = None
        self.id_map = []  # list of (db_id, name)

    def build_index(self):
        embeddings = get_all_embeddings()
        if not embeddings:
            # empty index; will be created on first add
            self.index = None
            self.id_map = []
            return

        vectors = []
        self.id_map = []
        for db_id, name, vec in embeddings:
            vectors.append(vec)
            self.id_map.append((db_id, name))

        matrix = np.vstack(vectors).astype(np.float32)
        dim = matrix.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(matrix)
        # try saving
        faiss.write_index(self.index, self.index_path)

    def add(self, db_id: int, name: str, vector: np.ndarray):
        vec = np.array(vector, dtype=np.float32).reshape(1, -1)
        if self.index is None:
            dim = vec.shape[1]
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(vec)
        self.id_map.append((db_id, name))
        faiss.write_index(self.index, self.index_path)

    def search(self, vector: np.ndarray, k=1):
        if self.index is None or self.index.ntotal == 0:
            return []
        vec = np.array(vector, dtype=np.float32).reshape(1, -1)
        D, I = self.index.search(vec, k)
        # D: distances; I: indices in id_map
        res = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.id_map):
                continue
            db_id, name = self.id_map[idx]
            res.append((db_id, name, float(dist)))
        return res