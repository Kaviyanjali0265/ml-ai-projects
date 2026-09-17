from pydantic import BaseModel, Field
from typing import List


class SearchInput(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=3, ge=1, le=10)


class SearchResult(BaseModel):
    id: str
    title: str
    tags: str
    content: str
    similarity_score: float


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
