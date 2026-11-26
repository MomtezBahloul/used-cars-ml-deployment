"""Model loading and prediction utilities for the 3-class used car classifier."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "used_cars_model.joblib"


def load_model(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    """Load the serialized scikit-learn pipeline bundle from disk."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {model_path}")
    return joblib.load(model_path)


def predict(bundle: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    """Run single-row inference and return native Python types."""
    input_df = pd.DataFrame([features], columns=bundle["columns"])
    pipeline = bundle["model"]
    label_encoder = bundle["label_encoder"]

    proba = pipeline.predict_proba(input_df)[0]
    model_classes = pipeline.classes_
    best_idx = int(np.argmax(proba))

    class_id = int(model_classes[best_idx])
    class_name = str(label_encoder.inverse_transform([class_id])[0])
    class_names = label_encoder.inverse_transform(model_classes)
    probabilities = {str(name): float(p) for name, p in zip(class_names, proba)}

    return {
        "class_id": class_id,
        "class_name": class_name,
        "probabilities": probabilities,
        "model_version": str(bundle["version"]),
    }
