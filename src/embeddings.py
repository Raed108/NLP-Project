from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


class ArabicEmbeddingModel:

    _model = None

    def __init__(self):

        if ArabicEmbeddingModel._model is None:

            ArabicEmbeddingModel._model = SentenceTransformer(
                EMBEDDING_MODEL_NAME
            )

        self.model = ArabicEmbeddingModel._model

    def embed_text(self, text):

        return self.model.encode(
            text,
            convert_to_tensor=False
        )

    def embed_texts(self, texts):

        return self.model.encode(
            texts,
            convert_to_tensor=False
        )