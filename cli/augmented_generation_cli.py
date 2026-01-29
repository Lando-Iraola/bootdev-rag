import argparse

from lib.augmented_generation import rag

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            result = rag(query)
            print("Search Results:")
            for r in result["search_results"]:
                print(f"  - {r}")
            print("RAG Response:")
            print(result["rag_response"])
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()