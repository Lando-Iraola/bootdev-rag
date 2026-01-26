from chunked_semantic_search import SemanticSearch


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        movies = []
        for document in documents:
            self.document_map[document["id"]] = document
            movies.append(f"{document['title']}: {document['description']}")
