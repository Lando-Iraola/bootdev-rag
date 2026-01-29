import argparse
import math
from lib.multimodal_search import verify_image_embedding, image_search_command



def main(rag_summarize=None):
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser(
        "verify_image_embedding", help="Generate an embedding for a single image"
    )

    verify_image_embedding_parser.add_argument("image_path",  type=str, help="Path to the image to have its embeddings")

    image_search_embedding_parser = subparsers.add_parser(
        "image_search", help="Generate an embedding for a single image"
    )

    image_search_embedding_parser.add_argument("image_path", type=str, help="Path to the image to initiate search")

    args = parser.parse_args()
    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path)
        case "image_search":
            result = image_search_command(args.image_path)
            for i, r in enumerate(result, 1):
                print(f"{i}. {r['title']} (similarity: {math.floor(r['score'] * 10 ** 3) / 10**3})")
                print(f"   {r['description'][:100]} ...")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
