"""The served endpoint answers on a raw record (skipped until the final model exists)."""
import json

import pytest
from fastapi.testclient import TestClient

from src.config import MODELS_DIR, ROOT

pytestmark = pytest.mark.skipif(not (MODELS_DIR / "final_model.joblib").exists(),
                                reason="run the training scripts first")


def test_predict_endpoint():
    from src.serve import app
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    person = json.loads((ROOT / "examples" / "person.json").read_text())
    body = client.post("/predict", json=person).json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] in (0, 1)


def test_predict_rejects_invalid_input():
    from src.serve import app
    client = TestClient(app)
    assert client.post("/predict", json={"age": -3}).status_code == 422
