import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import xgboost as xgb

from .config import MODEL_DIR, FEATURES, FAULTS, SEED


# Create synthetic telecom training data.
# These are simulated measurements, not real carrier data.
def generate_fault(row):
    # Priority matters when multiple rules match.
    if row["packet_loss"] >= 5:
        return "packet_loss"

    if row["latency"] >= 100:
        return "high_latency"

    if row["cpu"] >= 85:
        return "high_cpu"

    if row["traffic"] >= 90:
        return "congestion"

    return "normal"


def generate_dataset(n=2500):
    rng = np.random.default_rng(SEED)

    df = pd.DataFrame(
        {
            "cpu": rng.uniform(10, 100, n),
            "traffic": rng.uniform(10, 100, n),
            "packet_loss": rng.uniform(0, 10, n),
            "latency": rng.uniform(10, 180, n),
            "signal_strength": rng.uniform(-115, -50, n),
            "throughput": rng.uniform(1, 150, n),
        }
    )

    # Add a relationship between congestion and throughput.
    congested = df["traffic"] >= 90
    df.loc[congested, "throughput"] *= 0.55

    # Generate demonstration labels using transparent rules.
    df["fault"] = df.apply(generate_fault, axis=1)

    # Small label noise makes the toy example less trivial.
    # It does NOT make this dataset representative of real networks.
    noise_indices = rng.choice(n, size=int(n * 0.02), replace=False)

    for index in noise_indices:
        df.loc[index, "fault"] = rng.choice(FAULTS)

    return df


def train_model():
    df = generate_dataset()
    df.to_csv(Path(__file__).resolve().parent.parent / "telecom_training.csv", index=False)

    X = df[FEATURES]
    y = df["fault"]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=SEED,
        stratify=y_encoded,
    )

    model = xgb.XGBClassifier(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=SEED,
        n_jobs=2,
    )
    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)
    print("\n========== MODEL EVALUATION ==========")
    print("Test accuracy:", round(accuracy_score(y_test, test_predictions), 4))
    print(
        classification_report(
            y_test,
            test_predictions,
            labels=list(range(len(label_encoder.classes_))),
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_DIR / "xgb_model.json"))
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")
    print("Trained XGBoost model saved.")
