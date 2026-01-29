import argparse
import json
from lib.search_utils import GOLDEN_DATASET_PATH, RRF_K
from lib.hybrid_search import rrf_search_command


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        data_set = json.load(f)

    test_runs = []
    for test in data_set.get("test_cases"):
        results = rrf_search_command(test["query"], RRF_K, None, None, limit)
        relevant = 0
        for movie in results.get("results"):
            if movie.get("title") in test["relevant_docs"]:
                relevant += 1

        precision = relevant / len(results.get("results"))
        recall = relevant / len(test["relevant_docs"])
        run = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * (precision * recall) / (precision + recall),
            "query": test["query"],
            "retrieved": [m["title"] for m in results.get("results")],
            "relevant": test["relevant_docs"],
        }
        test_runs.append(run)

    print(f"k={limit}")
    for run in test_runs:
        print(f"  - Query: {run['query']}")
        print(f"    - Precision@{limit}: {run['precision']:.4f}")
        print(f"    - Recall@{limit}: {run['recall']:.4f}")
        print(f"    - F1 Score: {run['f1']:.4f}")
        print(f"    - Retrieved: {', '.join(run['retrieved'])}")
        print(f"    - Relevant: {', '.join(run['relevant'])}")

    # run evaluation logic here


if __name__ == "__main__":
    main()
