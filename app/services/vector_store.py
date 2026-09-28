from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


class VectorStore:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks: List[str] = []

    def build(self, chunks: List[str]) -> None:
        """
        Create embeddings for chunks and build a FAISS index.
        """

        if not chunks:
            raise ValueError("Cannot build vector store from empty chunks.")

        self.chunks = chunks

        embeddings = self.model.encode(
            chunks,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        embeddings = embeddings.astype("float32")

        dimension = embeddings.shape[1]

        # Inner product + normalized embeddings = cosine similarity
        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve the most relevant chunks for a query.
        """

        if self.index is None:
            raise RuntimeError("Vector store has not been built yet.")

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        query_embedding = query_embedding.astype("float32")

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, len(self.chunks)),
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            results.append(
                (
                    self.chunks[index],
                    float(score),
                )
            )

        return results

    def save(self, name: str) -> str:
        """
        Save FAISS index and chunks to data/<name>/.
        """

        if self.index is None:
            raise RuntimeError("Cannot save an empty vector store.")

        store_dir = DATA_DIR / name
        store_dir.mkdir(parents=True, exist_ok=True)

        index_path = store_dir / "index.faiss"
        chunks_path = store_dir / "chunks.npy"

        faiss.write_index(self.index, str(index_path))

        np.save(
            chunks_path,
            np.array(self.chunks, dtype=object),
            allow_pickle=True,
        )

        return str(store_dir)

    def load(self, name: str) -> None:
        """
        Load a previously saved vector store.
        """

        store_dir = DATA_DIR / name

        index_path = store_dir / "index.faiss"
        chunks_path = store_dir / "chunks.npy"

        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"Vector store not found at {store_dir}"
            )

        self.index = faiss.read_index(str(index_path))

        self.chunks = np.load(
            chunks_path,
            allow_pickle=True,
        ).tolist()