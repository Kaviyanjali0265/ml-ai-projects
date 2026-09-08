import os
import sys
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluate import get_shap_top_features
from preprocess import load_feature_names, preprocess_single
from api.schemas import CustomerInput, PredictionOutput

MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURE_NAMES_PATH = "models/feature_names.json"

model = None
scaler = None
feature_names = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, feature_names
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = load_feature_names(FEATURE_NAMES_PATH)
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts the probability that a telecom customer will churn, with SHAP-based explanations.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    customer_dict = customer.model_dump()
    input_array = preprocess_single(customer_dict, scaler, feature_names)

    churn_probability = float(model.predict_proba(input_array)[0][1])
    prediction = int(model.predict(input_array)[0])

    if churn_probability >= 0.7:
        risk_tier = "High"
    elif churn_probability >= 0.4:
        risk_tier = "Medium"
    else:
        risk_tier = "Low"

    top_factors = get_shap_top_features(model, input_array, feature_names, top_n=5)

    return PredictionOutput(
        churn_probability=round(churn_probability, 4),
        prediction=prediction,
        risk_tier=risk_tier,
        top_factors=top_factors,
    )
