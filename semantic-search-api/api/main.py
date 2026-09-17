import sys
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from api.schemas import SearchInput, SearchResponse, SearchResult
from src.indexer import index_knowledge_base
from src.searcher import search


@asynccontextmanager
async def lifespan(app: FastAPI):
    index_knowledge_base()
    yield


app = FastAPI(
    title="Semantic Search API",
    description="Search a tech knowledge base by meaning, not just keywords, using embeddings and ChromaDB.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/search", response_model=SearchResponse)
def semantic_search(input: SearchInput):
    results = search(query=input.query, top_k=input.top_k)
    return SearchResponse(
        query=input.query,
        results=[SearchResult(**r) for r in results]
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
