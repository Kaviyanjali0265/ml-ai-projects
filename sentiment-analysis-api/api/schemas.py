from pydantic import BaseModel, Field


class SentimentInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class SentimentOutput(BaseModel):
    sentiment: str
    confidence: float
    positive_score: float
    negative_score: float
