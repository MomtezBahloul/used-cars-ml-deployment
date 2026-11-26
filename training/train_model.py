"""Standalone training pipeline for 3-class used car price classification."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "used_cars_2026_dirty.csv"
MODEL_PATH = BASE_DIR / "models" / "used_cars_model.joblib"
INPUT_JSON_PATH = BASE_DIR / "input.json"

MODEL_VERSION = "1.0.0"

NUMERIC_COLS = [
    "registration_year",
    "mileage_km",
    "horsepower_ps",
    "engine_size_l",
    "previous_owners",
]
CATEGORICAL_COLS = [
    "brand",
    "vehicle_type",
    "fuel_type",
    "transmission",
    "unrepaired_damage",
]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS
IQR_COLS = [
    "registration_year",
    "mileage_km",
    "horsepower_ps",
    "engine_size_l",
]
INT_COLS = [
    "registration_year",
    "mileage_km",
    "horsepower_ps",
    "previous_owners",
]


def load_raw_dataset(data_path: Path) -> pd.DataFrame:
    """Load the raw CSV dataset or fail clearly if it is missing."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Fichier de données introuvable : {data_path}. "
            "Placez 'used_cars_2026_dirty.csv' dans le dossier 'data/'."
        )
    return pd.read_csv(data_path)


def remove_iqr_outliers(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Filter out rows outside [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR] for given columns."""
    mask = pd.Series(True, index=df.index)
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask &= df[col].between(lower, upper)
    return df.loc[mask].copy()


def clean_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Clean messy listings, impute missing values, and filter IQR outliers."""
    df = raw_df.drop_duplicates().copy()
    df = df.drop(columns=[c for c in ("listing_id", "currency") if c in df.columns])

    for col in ("mileage_km", "horsepower_ps"):
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"[^0-9.\-]", "", regex=True)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("registration_year", "engine_size_l", "previous_owners"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_COLS:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"nan": np.nan, "": np.nan})
        )

    df["price_category"] = (
        df["price_category"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": np.nan, "": np.nan})
    )

    if "listing_date" in df.columns:
        df["listing_date"] = pd.to_datetime(
            df["listing_date"],
            format="mixed",
            dayfirst=True,
            errors="coerce",
        )
        df = df.drop(columns=["listing_date"])

    df = df.dropna(subset=["brand", "price_category"]).copy()

    df["registration_year"] = df["registration_year"].fillna(
        round(float(df["registration_year"].mean()))
    )
    df["engine_size_l"] = df["engine_size_l"].fillna(
        round(float(df["engine_size_l"].mean()), 1)
    )

    for col in ("mileage_km", "horsepower_ps"):
        df[col] = df[col].fillna(df[col].median())

    for col in ("previous_owners", "fuel_type", "transmission"):
        mode_series = df[col].mode(dropna=True)
        if mode_series.empty:
            raise ValueError(f"Impossible de calculer le mode pour la colonne '{col}'.")
        df[col] = df[col].fillna(mode_series.iloc[0])

    for col in ("vehicle_type", "unrepaired_damage"):
        df[col] = df[col].fillna("unknown")

    df = remove_iqr_outliers(df, IQR_COLS)

    for col in INT_COLS:
        df[col] = df[col].round().astype(int)
    df["engine_size_l"] = df["engine_size_l"].round(1).astype(float)

    return df


def build_pipeline() -> Pipeline:
    """Create the scikit-learn preprocessing and LogisticRegression pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def export_sample_inputs(x_test: pd.DataFrame, output_path: Path) -> None:
    """Export two valid sample records from X_test matching API input bounds."""
    valid_samples = x_test[
        x_test["registration_year"].between(2010, 2026)
        & x_test["mileage_km"].between(0, 300000)
        & x_test["horsepower_ps"].between(50, 300)
        & x_test["engine_size_l"].between(0.0, 4.0)
        & x_test["previous_owners"].between(1, 5)
    ]
    sample_df = valid_samples.head(2) if len(valid_samples) >= 2 else x_test.head(2)
    sample_df.to_json(output_path, orient="records", indent=2)


def train_and_save() -> None:
    """Run end-to-end training, evaluation, serialization, and sample export."""
    raw_df = load_raw_dataset(DATA_PATH)
    cleaned_df = clean_dataset(raw_df)

    features = cleaned_df[FEATURE_COLS]
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(cleaned_df["price_category"])

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f} (samples={len(x_test)})")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": pipeline,
        "label_encoder": label_encoder,
        "columns": list(x_train.columns),
        "version": MODEL_VERSION,
        "sklearn_version": sklearn.__version__,
        "accuracy_test": round(float(accuracy), 4),
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)
    print(f"Saved model bundle to: {MODEL_PATH}")

    export_sample_inputs(x_test, INPUT_JSON_PATH)
    print(f"Exported sample inputs to: {INPUT_JSON_PATH}")


if __name__ == "__main__":
    train_and_save()
