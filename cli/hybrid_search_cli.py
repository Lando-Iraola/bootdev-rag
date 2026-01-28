import argparse
from lib.hybrid_search import HybridSearch, normalize, rrf_search_command
from lib.search_utils import load_movies
import os
from dotenv import load_dotenv
from google import genai
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser(
        "normalize", help="Normalizes scores to 0-1 range"
    )
    normalize_parser.add_argument(
        "list", nargs="+", type=float, help="Scores to normalize"
    )

    weighted_search_parser = subparser.add_parser(
        "weighted-search", help="Normalizes scores to 0-1 range"
    )

    weighted_search_parser.add_argument("query", type=str, help="Query")
    weighted_search_parser.add_argument(
        "--alpha", type=float, default=0.5, help="Alpha"
    )
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="Limit")

    rrf_search_parser = subparser.add_parser("rrf-search", help="Ranks results")
    rrf_search_parser.add_argument("query", type=str, help="Query")
    rrf_search_parser.add_argument("--k", type=int, default=60, help="Weight")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="Limit")
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Query reranking method",
    )
    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = normalize(args.list)
            for score in scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            hs = HybridSearch(load_movies())
            results = hs.weighted_search(args.query, args.alpha, args.limit)
            results = results[: args.limit]

            for index, result in enumerate(results):
                print(f"{index + 1} {result[1]['title']}")
                print(f"Hybrid score: {result[1]['hybrid']:.4f}")
                print(
                    f"BM25: {result[1]['bm25']:.4f}, Semantic: {result[1]['semantic']:.4f}"
                )
                print(result[1]["document"][:100])
        case "rrf-search":
            result = rrf_search_command(
                args.query, args.k, args.enhance, args.rerank_method, args.limit
            )

            if result["enhanced_query"]:
                print(
                    f"Enhanced query ({result['enhance_method']}): '{result['original_query']}' -> '{result['enhanced_query']}'\n"
                )

            if result["reranked"]:
                print(
                    f"Reranking top {len(result['results'])} results using {result['rerank_method']} method...\n"
                )

            print(
                f"Reciprocal Rank Fusion Results for '{result['query']}' (k={result['k']}):"
            )

            for i, res in enumerate(result["results"], 1):
                print(f"{i}. {res['title']}")
                if "individual_score" in res:
                    print(f"   Rerank Score: {res.get('individual_score', 0):.3f}/10")
                if "batch_score" in res:
                    print(f"   Rerank Rank: {res.get('batch_score')}")
                if "batch_rank" in res:
                    print(f"   Rerank Rank: {res.get('batch_rank', 0)}")
                if "cross_encoder_score" in res:
                    print(f"   Cross Encoder Score: {res.get('cross_encoder_score', 0)}")
                print(f"   RRF Score: {res.get('score', 0):.3f}")
                metadata = res.get("metadata", {})
                ranks = []
                if metadata.get("bm25_rank"):
                    ranks.append(f"BM25 Rank: {metadata['bm25_rank']}")
                if metadata.get("semantic_rank"):
                    ranks.append(f"Semantic Rank: {metadata['semantic_rank']}")
                if ranks:
                    print(f"   {', '.join(ranks)}")
                print(f"   {res['document'][:100]}...")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
