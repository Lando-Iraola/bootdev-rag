#!/usr/bin/env python3

import argparse
import math
import string

from inverted_index import InvertedIndex
from nltk.stem import PorterStemmer
from search_utils import BM25_B, BM25_K1

stemmer = PorterStemmer()


def tokenize(text):
    punctuation_map = {}
    for p in string.punctuation:
        punctuation_map[p] = None
    no_punctuation_table = str.maketrans(punctuation_map)

    with open(
        "./data/stopwords.txt",
        "r",
    ) as stop:
        stop_words = stop.read().splitlines()
    tokens = text.lower().translate(no_punctuation_table).split()
    tokens = [item for item in tokens if item not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = set()
    for token in tokens:
        stemmed_tokens.add(stemmer.stem(token))
    return list(stemmed_tokens)


def search_for_keyword(i_index, keyword=""):
    if not keyword:
        return []

    tokens = tokenize(keyword)
    results = []
    for token in tokens:
        for id in i_index.get_document(token):
            results.append(id)
            if len(results) == 5:
                break

        if len(results) == 5:
            break

    movie_descriptions = []

    for result in results:
        movie_descriptions.append(i_index.docmap[result])

    return movie_descriptions


def calculate_document_frequency(i_index, term):
    docs = i_index.get_document(term)
    return math.log((len(i_index.docmap) + 1) / (len(docs) + 1))


def calculate_tf_idf(i_index, movie_id, term):
    return i_index.term_frequencies[movie_id][
        tokenize(term)[0]
    ] * calculate_document_frequency(i_index, term)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    build_parser = subparsers.add_parser("build", help="Builds the search tokens")

    tf_parser = subparsers.add_parser(
        "tf",
        help="Term frequency is returned for a given movie ID and singular search term",
    )
    tf_parser.add_argument("movie_id", type=int, help="Movie ID")
    tf_parser.add_argument("movie_term", type=str, help="Singular frequency term")

    idf_parser = subparsers.add_parser(
        "idf", help="Document Frequency is return for a given term"
    )
    idf_parser.add_argument("movie_term", type=str, help="Singular frequency term")

    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Term frequency and inverse document frequency"
    )
    tfidf_parser.add_argument("movie_id", type=int, help="Movie ID")
    tfidf_parser.add_argument("movie_term", type=str, help="Singular frequency term")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )

    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )
    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            ii = InvertedIndex()
            try:
                ii.load()
            except Exception as e:
                print("couldn't search, missing index")
                print(e)
                exit(1)

            results = search_for_keyword(ii, args.query)
            count = 1
            results = sorted(results, key=lambda movie: movie["id"])
            for movie in results:
                print(f"{count}. Movie {movie['title']} {movie['id']}")
                count += 1
                if count > 5:
                    break
        case "build":
            ii = InvertedIndex()
            ii.build()
            ii.save()
        case "tf":
            id = args.movie_id
            term = args.movie_term
            ii = InvertedIndex()

            try:
                ii.load()
            except Exception as e:
                print("couldn't search, missing index")
                print(e)
                exit(1)

            print(ii.get_tf(id, term))
        case "idf":
            term = args.movie_term
            ii = InvertedIndex()

            try:
                ii.load()
            except Exception as e:
                print("couldn't search, missing index")
                print(e)
                exit(1)

            print(
                f"Inverse document frequency of '{term}': {
                    calculate_document_frequency(ii, term):.2f}"
            )
        case "tfidf":
            id = args.movie_id
            term = args.movie_term

            ii = InvertedIndex()

            try:
                ii.load()
            except Exception as e:
                print("couldn't search, missing index")
                print(e)
                exit(1)

            index = calculate_tf_idf(ii, id, term)
            print(f"TF-IDF score of '{term}' in document '{id}': {index:.2f}")
        case "bm25idf":
            term = args.term

            ii = InvertedIndex()

            try:
                ii.load()
            except Exception as e:
                print("couldn't search, missing index")
                print(e)
                exit(1)

            print(f"BM25 IDF score of '{term}': {ii.get_bm25_idf(term):.2f}")
        case "bm25tf":
            term = args.term
            id = args.doc_id
            k1 = args.k1
            b = args.b
            ii = InvertedIndex()

            try:
                ii.load()
            except Exception as e:
                print("couldn't search, missing index")
                print(e)
                exit(1)
            print(
                f"BM25 TF score of '{term}' in document '{id}': {
                    ii.get_bm25_tf(id, term, k1, b):.2f}"
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
