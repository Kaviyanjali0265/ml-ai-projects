import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from transformers import pipeline

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from api.schemas import SentimentInput, SentimentOutput

MODEL_PATH = "models/sentiment-model"

classifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    classifier = pipeline(
        "sentiment-analysis",
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        truncation=True,
        max_length=256,
    )
    yield


app = FastAPI(
    title="Sentiment Analysis API",
    description="Classifies text as Positive or Negative using fine-tuned DistilBERT on IMDB reviews.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": classifier is not None}


@app.post("/predict", response_model=SentimentOutput)
def predict(input: SentimentInput):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    raw = classifier(input.text, top_k=None)
    scores_list = raw[0] if isinstance(raw[0], list) else raw

    scores = {item["label"]: item["score"] for item in scores_list}
    positive_score = scores.get("POSITIVE", 0.0)
    negative_score = scores.get("NEGATIVE", 0.0)

    sentiment = "POSITIVE" if positive_score > negative_score else "NEGATIVE"
    confidence = max(positive_score, negative_score)

    return SentimentOutput(
        sentiment=sentiment,
        confidence=round(confidence, 4),
        positive_score=round(positive_score, 4),
        negative_score=round(negative_score, 4),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
