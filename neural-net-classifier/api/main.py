import os
import sys
from contextlib import asynccontextmanager

import joblib
import torch
from fastapi import FastAPI, HTTPException

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from model import LoanNet
from preprocess import load_feature_names, preprocess_single
from api.schemas import LoanInput, PredictionOutput

MODEL_PATH = "models/model.pt"
SCALER_PATH = "models/scaler.pkl"
FEATURE_NAMES_PATH = "models/feature_names.json"

model = None
scaler = None
feature_names = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, feature_names
    feature_names = load_feature_names(FEATURE_NAMES_PATH)
    scaler = joblib.load(SCALER_PATH)
    net = LoanNet(input_dim=len(feature_names))
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    net.to(device)
    net.eval()
    model = net
    yield


app = FastAPI(
    title="Loan Approval Prediction API",
    description="Predicts loan approval probability using a PyTorch neural network.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(loan: LoanInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    loan_dict = loan.model_dump()
    input_array = preprocess_single(loan_dict, scaler, feature_names)
    input_tensor = torch.FloatTensor(input_array).to(device)

    with torch.no_grad():
        prob = model(input_tensor).item()

    prediction = int(prob >= 0.5)
    decision = "Approved" if prediction == 1 else "Rejected"

    return PredictionOutput(
        approval_probability=round(prob, 4),
        prediction=prediction,
        decision=decision,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
