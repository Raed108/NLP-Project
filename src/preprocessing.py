import re
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter


def normalize_arabic(text):
    
    # 1. Convert English letters to lowercase
    text = text.lower()
    
    # 2. Remove Arabic Tashkeel (diacritics)
    tashkeel = r'[\u0617-\u061A\u064B-\u0652]'
    text = re.sub(tashkeel, '', text)

    # 3. Normalize Arabic letters
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    # text = re.sub("ؤ", "و", text)
    # text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)

    # 4. Remove Tatweel (ـ)
    text = re.sub("ـ", "", text)

    return text



def load_transcripts(folder_path):
    documents = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):

            path = os.path.join(folder_path, file_name)

            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            normalized = normalize_arabic(text)

            documents.append({
                "episode": file_name,
                "text": normalized
            })

    return documents


def chunk_documents(documents,
                    chunk_size=500,
                    chunk_overlap=100):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "؟", "!", " "]
    )

    chunks = []

    for doc in documents:

        split_texts = splitter.split_text(doc["text"])

        for idx, chunk in enumerate(split_texts):

            chunks.append({
                "text": chunk,
                "metadata": {
                    "episode": doc["episode"],
                    "chunk_id": idx
                }
            })

    return chunks