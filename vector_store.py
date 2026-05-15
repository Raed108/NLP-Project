from langchain_community.vectorstores import FAISS

from src.retrieval import CustomEmbeddings
from src.preprocessing import (
    load_transcripts,
    chunk_documents
)


def build_vector_store():

    documents = load_transcripts(
        "data/transcripts"
    )

    chunks = chunk_documents(documents)

    texts = [chunk["text"] for chunk in chunks]

    metadatas = [
        chunk["metadata"]
        for chunk in chunks
    ]

    embedding = CustomEmbeddings()

    db = FAISS.from_texts(
        texts=texts,
        embedding=embedding,
        metadatas=metadatas
    )

    db.save_local("faiss_index")

    print("FAISS index saved.")


if __name__ == "__main__":

    build_vector_store()