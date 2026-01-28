import argparse
from lib.hybrid_search import HybridSearch, normalize
from lib.search_utils import load_movies


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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
