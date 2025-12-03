#!/usr/bin/env python3

import argparse
import json

from lib import semantic_search


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verify_parser = subparsers.add_parser(
        "verify", help="Outputs current embeddings model"
    )
    embed_parser = subparsers.add_parser(
        "embed_text", help="Outputs current embeddings model"
    )
    embed_parser.add_argument("text", type=str, help="A word to get embeddings of")
    verify_embedding = subparsers.add_parser(
        "verify_embeddings", help="Verifies the embeddings"
    )
    embed_query_parser = subparsers.add_parser(
        "embedquery", help="Queries the embeddings"
    )
    embed_query_parser.add_argument("query", type=str, help="User query for movies")
    search_query_parser = subparsers.add_parser("search", help="Queries the embeddings")
    search_query_parser.add_argument("query", type=str, help="User query for movies")

    search_query_parser.add_argument(
        "--limit", type=int, default=5, help="Limit results to this number"
    )
    chunk_parser = subparsers.add_parser("chunk", help="Splits text into chunks")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")

    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="Size limit for chunks"
    )

    chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="Size limit for chunks"
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
        case "chunk":
            text = args.text.strip().split()
            chunks = []
            buffer = ""
            added_overlap = False
            for i, word in enumerate(text):
                if buffer == "":
                    buffer = word
                    continue
                if len(buffer.split()) < args.chunk_size or (
                    args.overlap > 0
                    and len(buffer.split()) < args.chunk_size + args.overlap
                    and len(chunks) > 0
                ):
                    if args.overlap > 0 and len(chunks) > 0 and not added_overlap:
                        overlap = args.overlap * -1
                        last_two = chunks[-1:][0].split()[overlap:]
                        last_two = " ".join(last_two)
                        buffer = f"{last_two} {buffer}"
                        added_overlap = True
                    buffer = f"{buffer} {word}"
                else:
                    chunks.append(buffer)
                    buffer = word
                    added_overlap = False

                if i + 1 == len(text):
                    chunks.append(buffer)

            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks):
                print(f"{i + 1}. {chunk}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
