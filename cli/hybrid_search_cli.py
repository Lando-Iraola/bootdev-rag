import argparse
from lib.hybrid_search import HybridSearch, normalize
from lib.search_utils import load_movies
import os
from dotenv import load_dotenv
from google import genai


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
            hs = HybridSearch(load_movies())
            if args.enhance:
                load_dotenv()
                api_key = os.environ.get("GEMINI_API_KEY")
                client = genai.Client(api_key=api_key)
                prompt = ""
                match args.enhance:
                    case "spell":
                        prompt = f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{args.query}"

If no errors, return the original query.
Corrected:"""
                    case "rewrite":
                        prompt = f"""Rewrite this movie search query to be more specific and searchable.

Original: "{args.query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""
                    case "expand":
                        prompt = f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{args.query}"
"""
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )

                print(
                    f"Enhanced query ({args.enhance}): '{args.query}' -> '{response.text}'\n"
                )
                args.query = response.text
            results = hs.rrf_search(args.query, args.k, args.limit)
            results = results[: args.limit]

            for index, result in enumerate(results):
                print(f"{index + 1} {result[1]['title']}")
                print(f"RRF score: {result[1]['rank']:.4f}")
                print(
                    f"BM25: {result[1]['bm25']:.4f}, Semantic: {result[1]['semantic']:.4f}"
                )
                print(result[1]["document"][:100])
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
