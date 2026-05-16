import numpy as np

from src.embeddings import ArabicEmbeddingModel


class SemanticCache:

    def __init__(self,
                 threshold=0.85):

        self.threshold = threshold

        self.embedding_model = ArabicEmbeddingModel()

        self.cache = []
        self.max_cache_size = 100
        self.saved_calls = 0
        self.cache_hits = 0
        self.total_requests = 0

    def normalize_vector(self, vector):

        norm = np.linalg.norm(vector)

        if norm == 0:
            return vector

        return vector / norm

    def cosine_similarity(self,
                          vec1,
                          vec2):

        return np.dot(vec1, vec2)

    def search_cache(self,
                     query):

        self.total_requests += 1

        query_embedding = self.embedding_model.embed_text(query)

        query_embedding = self.normalize_vector(
            query_embedding
        )

        best_similarity = -1
        best_response = None

        for item in self.cache:

            similarity = self.cosine_similarity(
                query_embedding,
                item["embedding"]
            )

            if similarity > best_similarity:

                best_similarity = similarity
                best_response = item["response"]

        if best_similarity >= self.threshold:

            self.cache_hits += 1
            self.saved_calls += 1

            return {
                "hit": True,
                "response": best_response,
                "similarity": float(best_similarity)
            }

        return {
            "hit": False,
            "similarity": float(best_similarity)
        }

    def add_to_cache(self,
                     question,
                     response):

        for item in self.cache:

            if item["question"] == question:

                return

        embedding = self.embedding_model.embed_text(question)

        embedding = self.normalize_vector(embedding)

        if len(self.cache) >= self.max_cache_size:
            self.cache.pop(0)
            
        self.cache.append({
            "question": question,
            "response": response,
            "embedding": embedding
        })

    def hit_rate(self):

        if self.total_requests == 0:

            return 0

        return self.cache_hits / self.total_requests
    
    def saved_model_calls(self):

        return self.saved_calls