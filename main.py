"""
main.py
FastAPI application that serves the Iris classification pipeline.

Run:  fastapi dev main.py
Docs: http://localhost:8000/docs
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "model" / "pipeline.pkl"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# ---------------------------------------------------------------------------
# Load the model ONCE at startup (not inside the predict function)
# ---------------------------------------------------------------------------
if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model file not found at {MODEL_PATH}. Run 'python save_model.py' first."
    )
pipeline = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Iris Flower Classification API",
    description="Predicts the species of an Iris flower (setosa, versicolor, "
                "virginica) from its sepal and petal measurements.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class InputData(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm", examples=[5.1])
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm", examples=[3.5])
    petal_length: float = Field(..., gt=0, description="Petal length in cm", examples=[1.4])
    petal_width: float = Field(..., gt=0, description="Petal width in cm", examples=[0.2])


class PredictionOutput(BaseModel):
    prediction: str
    probability: float
    all_probabilities: dict[str, float]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "name": "Iris Flower Classification API",
        "description": "Send flower measurements to POST /predict to get the "
                       "predicted Iris species. See /docs for details.",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: InputData):
    try:
        # DataFrame with the same column names used during training
        features = pd.DataFrame([data.model_dump()], columns=FEATURES)

        pred = pipeline.predict(features)[0]
        proba = pipeline.predict_proba(features)[0]

        return PredictionOutput(
            prediction=str(pred),
            probability=round(float(max(proba)), 4),
            all_probabilities={
                str(cls): round(float(p), 4)
                for cls, p in zip(pipeline.classes_, proba)
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
