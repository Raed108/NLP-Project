import re
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter


def normalize_arabic(text):
    
    if text is None:
        return ""

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

def clean_transcript(text: str):

    # Convert escaped newlines to real newlines
    text = text.replace("\\n", "\n")

    # Remove subtitle timestamps
    # Example: 1722.027:
    text = re.sub(r"\d+\.\d+:", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()



def load_transcripts(folder_path):

    documents = []

    # Walk through ALL directories recursively
    for root, dirs, files in os.walk(folder_path):

        for file_name in files:

            if file_name.endswith(".txt"):

                path = os.path.join(root, file_name)

                with open(path, "r", encoding="utf-8") as f:

                    text = f.read()

                # Clean transcript
                text = clean_transcript(text)

                # Normalize Arabic
                normalized = normalize_arabic(text)

                documents.append({
                    "episode": file_name,
                    "text": normalized
                })

    return documents


def chunk_documents(documents,
                    chunk_size=200,
                    chunk_overlap=50):

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