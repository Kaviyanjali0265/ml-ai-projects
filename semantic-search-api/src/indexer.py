import json
import os

import chromadb

from src.embedder import embed_batch

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base.json")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "tech_knowledge_base"


def get_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def index_knowledge_base():
    with open(DATA_PATH) as f:
        docs = json.load(f)

    collection = get_collection()

    if collection.count() == len(docs):
        print(f"Already indexed {len(docs)} documents. Skipping.")
        return

    texts = [f"{doc['title']}. {doc['content']}" for doc in docs]
    embeddings = embed_batch(texts)

    collection.upsert(
        ids=[doc["id"] for doc in docs],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"title": doc["title"], "tags": ", ".join(doc["tags"])} for doc in docs]
    )
    print(f"Indexed {len(docs)} documents into ChromaDB.")


if __name__ == "__main__":
    index_knowledge_base()
