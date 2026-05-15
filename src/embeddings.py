from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


class ArabicEmbeddingModel:

    def __init__(self):

        self.model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

    def embed_text(self, text):

        return self.model.encode(text)

    def embed_texts(self, texts):

        return self.model.encode(texts)