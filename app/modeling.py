from __future__ import annotations

import joblib
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .config import FEATURES, MODEL_DIR, SEED
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


def evaluate_model():
    if inference_model is None or inference_encoder is None:
        raise RuntimeError("The prediction model has not been initialized.")

    data_path = MODEL_DIR.parent / "telecom_training.csv"
    if not data_path.exists():
        raise RuntimeError(f"The model evaluation dataset was not found: {data_path}")

    dataset = pd.read_csv(data_path)
    required_columns = [*FEATURES, "fault"]
    missing_columns = [column for column in required_columns if column not in dataset.columns]
    if missing_columns:
        raise RuntimeError(
            "The model evaluation dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    features = dataset[FEATURES]
    try:
        labels = inference_encoder.transform(dataset["fault"])
    except ValueError as error:
        raise RuntimeError("The evaluation dataset contains unknown fault labels.") from error

    _, test_features, _, test_labels = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=SEED,
        stratify=labels,
    )
    predictions = inference_model.predict(test_features)
    class_labels = list(range(len(inference_encoder.classes_)))
    report = classification_report(
        test_labels,
        predictions,
        labels=class_labels,
        target_names=[str(label) for label in inference_encoder.classes_],
        output_dict=True,
        zero_division=0,
    )
    accuracy = float(accuracy_score(test_labels, predictions))

    return {
        "evaluation_dataset": data_path.name,
        "evaluation_method": "20% stratified holdout",
        "test_samples": len(test_labels),
        "accuracy": accuracy,
        "score": accuracy,
        "classification_report": report,
        "confusion_matrix": {
            "labels": [str(label) for label in inference_encoder.classes_],
            "values": confusion_matrix(
                test_labels,
                predictions,
                labels=class_labels,
            ).tolist(),
        },
    }
