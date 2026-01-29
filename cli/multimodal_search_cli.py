import argparse

from lib.multimodal_search import verify_image_embedding


def main(rag_summarize=None):
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser(
        "verify_image_embedding", help="Generate an embedding for a single image"
    )

    verify_image_embedding_parser.add_argument("image_path",  type=str, help="Path to the image to have its embeddings")

    args = parser.parse_args()
    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
