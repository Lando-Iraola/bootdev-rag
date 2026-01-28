import os

from .keyword_search import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_results = self._bm25_search(query, limit * 500)
        bm25_scores = [bm25["score"] for bm25 in bm25_results]
        normalized_bm25_scores = normalize(bm25_scores)

        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        semantic_scores = [semantic["score"] for semantic in semantic_results]
        normalized_semantic_scores = normalize(semantic_scores)

        mixed_scores = {}
        for index, score in enumerate(normalized_bm25_scores):
            if bm25_results[index]["id"] not in mixed_scores:
                mixed_scores[bm25_results[index]["id"]] = bm25_results[index]
                mixed_scores[bm25_results[index]["id"]]["bm25"] = score

        for index, score in enumerate(normalized_semantic_scores):
            mixed_scores[semantic_results[index]["id"]]["semantic"] = score

        for m_s in mixed_scores:
            mixed_scores[m_s]["hybrid"] = hybrid_score(
                mixed_scores[m_s]["bm25"], mixed_scores[m_s]["semantic"], alpha
            )

        sorted_scores = sorted(
            mixed_scores.items(), key=lambda x: x[1]["hybrid"], reverse=True
        )

        return sorted_scores

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize(list):
    if not list:
        return
    max_val = max(list)
    min_val = min(list)

    scores = []
    if max_val == min_val:
        for val in list:
            scores.append(1.0)
        return scores

    for val in list:
        scores.append((val - min_val) / (max_val - min_val))

    return scores


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score
