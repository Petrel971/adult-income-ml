"""Prediction endpoint: raw person record in, probability and decision out.

Run:   uvicorn src.serve:app --port 8000
Try:   curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
            -d @examples/person.json
       curl -X POST localhost:8000/predict/batch -H "Content-Type: application/json" \
            -d "[$(cat examples/person.json)]"
"""
from functools import lru_cache
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from src.config import MODELS_DIR

app = FastAPI(title="Adult income classifier", version="1.0")


class Person(BaseModel):
    """One census record. Field names follow the dataset (aliases keep the hyphens)."""
    model_config = ConfigDict(populate_by_name=True)

    age: int = Field(ge=16, le=100)
    workclass: Optional[str] = None
    education_num: int = Field(alias="education-num", ge=1, le=16)
    marital_status: str = Field(alias="marital-status")
    occupation: Optional[str] = None
    relationship: str
    race: str
    sex: str
    capital_gain: float = Field(0, alias="capital-gain", ge=0)
    capital_loss: float = Field(0, alias="capital-loss", ge=0)
    hours_per_week: int = Field(alias="hours-per-week", ge=1, le=99)
    native_country: Optional[str] = Field(None, alias="native-country")


class Prediction(BaseModel):
    probability: float
    prediction: int
    label: str
    threshold: float
    model: str


@lru_cache
def load_model():
    return joblib.load(MODELS_DIR / "final_model.joblib")


@app.get("/health")
def health():
    final = load_model()
    return {"status": "ok", "model": final["model_name"], "threshold": final["threshold"]}


@app.post("/predict", response_model=Prediction)
def predict(person: Person):
    final = load_model()
    row = pd.DataFrame([person.model_dump(by_alias=True)])[final["columns"]]
    proba = float(final["pipeline"].predict_proba(row)[0, 1])
    decision = int(proba >= final["threshold"])
    return Prediction(probability=round(proba, 4), prediction=decision,
                      label=">50K" if decision else "<=50K",
                      threshold=round(final["threshold"], 4), model=final["model_name"])


@app.post("/predict/batch", response_model=List[Prediction])
def predict_batch(people: List[Person]):
    """Several records in one call, scored with the same Pipeline and threshold."""
    return [predict(p) for p in people]
