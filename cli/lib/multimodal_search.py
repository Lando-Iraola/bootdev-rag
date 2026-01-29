from PIL import Image
from sentence_transformers import SentenceTransformer
from .semantic_search import cosine_similarity
from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies

class MultimodalSearch:
    def __init__(self,documents = list[dict], model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = [f"{d['title']}: {d['description']}" for d in self.documents]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
    def embed_image(self, path):
        img = Image.open(path)
        embedding = self.model.encode([img])[0]
        return embedding
    def search_with_image(self, image_path):
        image_embedding = self.embed_image(image_path)
        similarity = []
        for i, text_embedding in enumerate(self.text_embeddings):
            similarity.append((cosine_similarity(text_embedding, image_embedding), self.documents[i]))

        similarity.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in similarity[:DEFAULT_SEARCH_LIMIT]:
            results.append(
                {
                    "score": score,
                    "id": doc['id'],
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )

        return results

def verify_image_embedding(image_path):
    ms = MultimodalSearch()
    embedding = ms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path):
    ms = MultimodalSearch(load_movies())
    result = ms.search_with_image(image_path)
    return result
