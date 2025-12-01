#!/usr/bin/env python3

import argparse

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

    args = parser.parse_args()

    match args.command:
        case "verify":
            semantic_search.verify_model()
        case "embed_text":
            semantic_search.embed_text(args.text)
        case "verify_embeddings":
            semantic_search.verify_embeddings()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
