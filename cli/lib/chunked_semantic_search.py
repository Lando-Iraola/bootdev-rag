from .semantic_search import SemanticSearch, semantic_chunk, cosine_similarity
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
    format_chunked_search_result,
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
        for index, doc in enumerate(documents):
            text = doc.get("description", "")
            if not text.strip():
                continue

            chunks = semantic_chunk(
                text,
                max_chunk_size=DEFAULT_SEMANTIC_CHUNK_SIZE,
                overlap=DEFAULT_CHUNK_OVERLAP,
            )

            for c_index, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                these_chunks = {
                    "movie_idx": index,
                    "chunk_idx": c_index,
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

    def search_chunks(self, query: str, limit: int = 10):
        if not query:
            return

        embeddings = self.generate_embedding(query)
        chunk_score = []
        for index, chunk in enumerate(self.chunk_embeddings):
            score = cosine_similarity(chunk, embeddings)
            chunk_score.append(
                {
                    "chunk_idx": index,
                    "movie_idx": self.chunk_metadata['chunks'][index]['movie_idx'],
                    "score": score,
                }
            )

        movie_score = {}
        for c_score in chunk_score:
            if (
                c_score["movie_idx"] not in movie_score
                or c_score["score"] > movie_score[c_score["movie_idx"]]["score"]
            ):
                movie_score[c_score["movie_idx"]] = c_score
        sorted_scores = sorted(movie_score.items(), key=lambda x: x[1]['score'], reverse=True)
        sorted_scores = sorted_scores[:limit]
        results = []
        for doc in sorted_scores:
            formatted_result = format_chunked_search_result(
                doc_id=self.documents[doc[1]['movie_idx']]["id"],
                title=self.documents[doc[1]['movie_idx']]["title"],
                document=self.documents[doc[1]['movie_idx']]["description"],
                score=doc[1]['score'],
            )
            results.append(formatted_result)
        return results
