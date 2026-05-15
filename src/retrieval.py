from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

from src.embeddings import ArabicEmbeddingModel
import os

class CustomEmbeddings(Embeddings):

    def __init__(self):

        self.embedding_model = ArabicEmbeddingModel()

    def embed_documents(self, texts):

        return self.embedding_model.embed_texts(texts).tolist()

    def embed_query(self, text):

        return self.embedding_model.embed_text(text).tolist()


class RetrieverSystem:

    def __init__(self, db_path="faiss_index"):

        self.embedding = CustomEmbeddings()
        if os.path.exists(os.path.join(db_path, "index.faiss")):
            self.db = FAISS.load_local(
                db_path,
                self.embedding,
                allow_dangerous_deserialization=True
            )
        else:
            print("FAISS index not found, creating new DB...")
            self.db = FAISS.from_texts([], self.embedding)
            self.db.save_local(db_path)

    def retrieve(self, query, k=5):

        docs = self.db.similarity_search_with_score(query, k=k)

        return docs