from __future__ import annotations

import joblib
import pandas as pd
import xgboost as xgb
import numpy as np

from .config import FEATURES, MODEL_DIR
from .data_generation import train_model

inference_model: xgb.XGBClassifier | None = None
inference_encoder = None


def initialize_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "xgb_model.json"
    encoder_path = MODEL_DIR / "label_encoder.pkl"
    if not model_path.exists() or not encoder_path.exists():
        train_model()

    global inference_model, inference_encoder
    inference_model = xgb.XGBClassifier()
    inference_model.load_model(str(model_path))
    inference_encoder = joblib.load(encoder_path)


def predict_fault(metrics):
    if inference_model is None or inference_encoder is None:
        raise RuntimeError("The prediction model has not been initialized.")

    row = pd.DataFrame(
        [[metrics[name] for name in FEATURES]],
        columns=FEATURES,
    )

    probabilities = inference_model.predict_proba(row)[0]
    best_index = int(np.argmax(probabilities))

    return {
        "fault": str(inference_encoder.inverse_transform([best_index])[0]),
        "confidence": float(probabilities[best_index]),
        "probabilities": {
            str(label): float(probability)
            for label, probability in zip(inference_encoder.classes_, probabilities)
        },
    }
