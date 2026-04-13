from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "titanic_model.joblib"

app = FastAPI(title="Titanic CI/CD API", version="0.1.0")


class PredictRequest(BaseModel):
    pclass: int = Field(..., ge=1, le=3)
    sex: str
    age: float | None = None
    sibsp: int = Field(0, ge=0)
    parch: int = Field(0, ge=0)
    fare: float | None = None
    embarked: str | None = None


class PredictResponse(BaseModel):
    prediction: int
    label: str


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Run train.py first."
        )
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    model = load_model()

    df = pd.DataFrame(
        [
            {
                "Pclass": payload.pclass,
                "Sex": payload.sex,
                "Age": payload.age,
                "SibSp": payload.sibsp,
                "Parch": payload.parch,
                "Fare": payload.fare,
                "Embarked": payload.embarked,
            }
        ]
    )

    pred = int(model.predict(df)[0])
    label = "survived" if pred == 1 else "did_not_survive"

    return PredictResponse(prediction=pred, label=label)
