import argparse

from lib.augmented_generation import rag, summarize, citations, answer_question


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    rag_summarize = subparsers.add_parser(
        "summarize", help="Summarize the contents of documents found"
    )
    rag_summarize.add_argument("query", type=str, help="Search query")
    rag_summarize.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )

    rag_citations = subparsers.add_parser(
        "citations", help="Summarize the contents of documents found and offer citations to them"
    )
    rag_citations.add_argument("query", type=str, help="Search query")
    rag_citations.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )
    rag_question = subparsers.add_parser(
        "question", help="Answer user questions"
    )
    rag_question.add_argument("question", type=str, help="Search query")
    rag_question.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )

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
        case "summarize":
            query = args.query
            limit = args.limit
            result = summarize(query, limit)
            print("Search Results:")
            for r in result["search_results"]:
                print(f"  - {r}")
            print()
            print("LLM Summary:")
            print(result["summary"])
        case "citations":
            query = args.query
            limit = args.limit
            result = citations(query, limit)
            print("Search Results:")
            for i, r in enumerate(result["search_results"], 1):
                print(f"  {i}. {r[0]} - ID: {r[1]} ")
            print()
            print("LLM Answer:")
            print(result["citations"])
        case "question":
            question = args.question
            limit = args.limit
            result = answer_question(question, limit)
            print("Search Results:")
            for i, r in enumerate(result["search_results"], 1):
                print(f"  {i}. {r[0]} - ID: {r[1]} ")
            print()
            print("Answer:")
            print(result["answer"])
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
