import os

BM25_K1 = 1.5
BM25_B = 0.75
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()
