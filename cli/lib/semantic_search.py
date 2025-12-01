import json
import os

import numpy as np
from search_utils import EMBEDDINGS_PATH
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model = model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if not text or text.isspace():
            raise ValueError("Empty string")
        embeddings = self.model.encode([text])
        return embeddings[0]

    def build_embeddings(self, documents):
        self.documents = documents
        movies = []
        for document in documents:
            self.document_map[document["id"]] = document
            movies.append(f"'{document['title']}': {document['description']}")
        self.embeddings = self.model.encode(movies, show_progress_bar=True)
        np.save(EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document

        if not os.path.exists(EMBEDDINGS_PATH):
            return self.build_embeddings(documents)

        self.embeddings = np.load(EMBEDDINGS_PATH)

        if len(self.embeddings) != len(self.documents):
            return self.build_embeddings(documents)

        return self.embeddings


def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    ss = SemanticSearch()
    with open(
        "./data/movies.json",
        "r",
    ) as movies:
        data_set = json.load(movies)
    ss.load_or_create_embeddings(data_set["movies"])
    print(f"Number of docs:   {len(data_set['movies'])}")
    print(
        f"Embeddings shape: {ss.embeddings.shape[0]} vectors in {
            ss.embeddings.shape[1]
        } dimensions"
    )
