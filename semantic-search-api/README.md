# Semantic Search API

A semantic search engine over a tech knowledge base — finds articles by **meaning**, not just keywords, using sentence embeddings and ChromaDB as the vector store.

---

## What It Does

Given a natural language query, the API returns the most semantically similar articles from a tech knowledge base — even when the query shares no exact words with the result.

```json
POST /search

Query: "my containers cannot talk to each other"

{
  "query": "my containers cannot talk to each other",
  "results": [
    {
      "id": "kb003",
      "title": "Docker Container Networking",
      "tags": "docker, networking, devops",
      "content": "Docker containers communicate via bridge networks by default...",
      "similarity_score": 0.277
    }
  ]
}
```

No keyword match. Pure semantic understanding.

---

## Architecture

```
Startup (once):
  knowledge_base.json
        |
  nomic-embed-text (Ollama)   embeds each article
        |
  ChromaDB                    stores vectors on disk

Query (every request):
  User query
        |
  nomic-embed-text            embeds the query
        |
  ChromaDB                    nearest neighbor search
        |
  Top-K matching articles + similarity scores
```

---

## Key Concepts

**Embeddings** — dense vectors (768 numbers) representing the meaning of text. Similar meaning = similar vectors = small angle between them.

**Cosine similarity** - measures the angle between two vectors. 1.0 = identical meaning, 0.0 = unrelated.

**Vector database** - stores embeddings and does fast nearest neighbor search. ChromaDB runs locally with persistent storage on disk.

**Semantic vs keyword search** - keyword search matches exact words. Semantic search matches meaning. "containers cannot communicate" matches "Docker networking" even with zero word overlap.

---

## Dataset

15 tech knowledge base articles covering: databases, caching, Docker, Kubernetes, Python, REST APIs, CI/CD, microservices, Git, and more. Stored in `data/knowledge_base.json`.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Embedding Model | nomic-embed-text (Ollama) |
| Vector Database | ChromaDB |
| Embedding Client | OpenAI Python SDK (Ollama-compatible) |
| API | FastAPI + Pydantic |
| Server | Uvicorn |
| Containerization | Docker |
| Testing | Pytest |

---

## Project Structure

```
semantic-search-api/
├── src/
│   ├── embedder.py     # text to vector via nomic-embed-text
│   ├── indexer.py      # load knowledge base into ChromaDB
│   └── searcher.py     # query ChromaDB and return results
├── api/
│   ├── main.py         # FastAPI app
│   └── schemas.py      # Pydantic input/output schemas
├── data/
│   └── knowledge_base.json   # 15 tech articles
├── tests/
│   └── test_api.py
├── Dockerfile
├── Makefile
└── requirements.txt
```

---

## Setup and Run

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

> Requires [Ollama](https://ollama.com) running locally with `nomic-embed-text` pulled:
> ```bash
> ollama pull nomic-embed-text
> ollama serve
> ```

```bash
make serve
# Open http://localhost:8000/docs
```

**Run with Docker:**
```bash
make docker-build
make docker-run
```

**Run tests:**
```bash
make test
```

---

## Sample API Calls

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "how to handle slow database queries"}'
```

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "my containers cannot talk to each other", "top_k": 5}'
```
