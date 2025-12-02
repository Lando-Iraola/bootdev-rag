#!/usr/bin/env python3

import argparse
import json

from lib import semantic_search


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")
    verify_parser = subparsers.add_parser(
        "verify", help="Outputs current embeddings model"
    )
    embed_parser = subparsers.add_parser(
        "embed_text", help="Outputs current embeddings model"
    )
    embed_parser.add_argument(
        "text", type=str, help="A word to get embeddings of")
    verify_embedding = subparsers.add_parser(
        "verify_embeddings", help="Verifies the embeddings"
    )
    embed_query_parser = subparsers.add_parser(
        "embedquery", help="Queries the embeddings"
    )
    embed_query_parser.add_argument(
        "query", type=str, help="User query for movies")
    search_query_parser = subparsers.add_parser(
        "search", help="Queries the embeddings")
    search_query_parser.add_argument(
        "query", type=str, help="User query for movies")

    search_query_parser.add_argument(
        "--limit", type=int, default=5, help="Limit results to this number"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            semantic_search.verify_model()
        case "embed_text":
            semantic_search.embed_text(args.text)
        case "verify_embeddings":
            semantic_search.verify_embeddings()
        case "embedquery":
            ss = semantic_search.SemanticSearch()
            emb = ss.generate_embedding(args.query)
            print(f"Query: {args.query}")
            print(f"First 5 dimensions: {emb[:5]}")
            print(f"Shape: {emb.shape}")

        case "search":
            ss = semantic_search.SemanticSearch()
            with open(
                "./data/movies.json",
                "r",
            ) as movies:
                data_set = json.load(movies)

            ss.load_or_create_embeddings(data_set["movies"])
            res = ss.search(args.query, args.limit)
            for i, v in enumerate(res):
                print(f"{i + 1}. {v['title']} (score: {v['score']:.4f}")
                print(f"{v['description']}\n")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
