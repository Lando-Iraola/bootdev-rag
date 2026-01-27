from .semantic_search import SemanticSearch, semantic_chunk
import os
import numpy as np
import json
from .search_utils import (
    CACHE_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    load_movies,
)


MOVIE_CHUNKED_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
MOVIE_CHUNKED_META_DATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunks_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        all_chunks = []
        chunk_metadata = []
        for doc in documents:
            if not doc["description"]:
                continue
            chunks = semantic_chunk(doc["description"])
            all_chunks.extend(chunks)
            for chunk in chunks:
                these_chunks = {
                    "movie_idx": doc["id"],
                    "chunk_idx": chunks.index(chunk),
                    "total_chunks": len(chunks),
                }
                chunk_metadata.append(these_chunks)

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        os.makedirs(os.path.dirname(MOVIE_CHUNKED_EMBEDDINGS_PATH), exist_ok=True)
        np.save(MOVIE_CHUNKED_EMBEDDINGS_PATH, self.chunk_embeddings)

        with open(MOVIE_CHUNKED_META_DATA_PATH, "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)},
                f,
                indent=2,
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(MOVIE_CHUNKED_EMBEDDINGS_PATH) and os.path.exists(
            MOVIE_CHUNKED_META_DATA_PATH
        ):
            self.chunk_embeddings = np.load(MOVIE_CHUNKED_EMBEDDINGS_PATH)
            with open(MOVIE_CHUNKED_META_DATA_PATH, "r", encoding="utf-8") as f:
                self.chunk_metadata = json.load(f)
            return self.chunk_embeddings

        return self.build_chunks_embeddings(documents)
