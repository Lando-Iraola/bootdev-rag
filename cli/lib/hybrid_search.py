import os
from typing import Optional
from .keyword_search import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch
from .query_enhancement import enhance_query
from .reranking import rerank
from .search_utils import (
    DEFAULT_ALPHA,
    DEFAULT_SEARCH_LIMIT,
    RRF_K,
    SEARCH_MULTIPLIER,
    format_search_result,
    load_movies,
)


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
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        combined = combine_search_results(bm25_results, semantic_results, alpha)

        return combined[:limit]

    def rrf_search(self, query, k, limit=10):
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        fused = reciprocal_rank_fusion(bm25_results, semantic_results, k)
        return fused[:limit]


def combine_search_results(bm25_results, semantic_results, alpha):
    bm25_scores = [bm25["score"] for bm25 in bm25_results]
    normalized_bm25_scores = normalize(bm25_scores)

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

    hybrid_results = []
    for doc_id, data in mixed_scores.items():
        score_value = hybrid_score(data["bm25_score"], data["semantic_score"], alpha)
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=score_value,
            bm25_score=data["bm25"],
            semantic_score=data["semantic"],
        )
        hybrid_results.append(result)

    sorted_scores = sorted(hybrid_results, key=lambda x: x[1]["hybrid"], reverse=True)

    return sorted_scores


def reciprocal_rank_fusion(bm25_results, semantic_results, k):
    rrf = {}
    for index, bm25 in enumerate(bm25_results):
        if bm25["id"] not in rrf:
            rrf[bm25["id"]] = bm25.copy()
            del rrf[bm25["id"]]["score"]
            rrf[bm25["id"]]["semantic"] = 0
            rrf[bm25["id"]]["bm25"] = bm25["score"]
            rrf[bm25["id"]]["rank"] = rrf_score(index, k)

    for index, semantic in enumerate(semantic_results):
        if semantic["id"] in rrf:
            rrf[semantic["id"]]["rank"] += rrf_score(index, k)
        else:
            rrf[semantic["id"]] = semantic.copy()
            del rrf[semantic["id"]]["score"]
            if "bm25" not in rrf[semantic["id"]]:
                rrf[semantic["id"]]["bm25"] = 0
            rrf[semantic["id"]]["semantic"] = semantic["score"]
            rrf[semantic["id"]]["rank"] = rrf_score(index, k)

    sorted_scores = sorted(rrf.items(), key=lambda x: x[1]["rank"], reverse=True)
    sorted_scores = [item[1] for item in sorted_scores]
    return sorted_scores


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


def rrf_score(rank, k=60):
    return 1 / (k + rank)


def rrf_search_command(
    query: str,
    k: int = RRF_K,
    enhance: Optional[str] = None,
    rerank_method: Optional[str] = None,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)
    original_query = query
    enhanced_query = None
    if enhance:
        enhanced_query = enhance_query(query, method=enhance)
        query = enhanced_query

    search_limit = limit * SEARCH_MULTIPLIER if rerank_method else limit
    results = searcher.rrf_search(query, k, search_limit)

    reranked = False
    if rerank_method:
        results = rerank(query, results, method=rerank_method, limit=limit)
        reranked = True

    return {
        "original_query": original_query,
        "enhanced_query": enhanced_query,
        "enhance_method": enhance,
        "query": query,
        "k": k,
        "rerank_method": rerank_method,
        "reranked": reranked,
        "results": results,
    }
