import os
from openai import OpenAI

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


def embed(text: str) -> list[float]:
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]
