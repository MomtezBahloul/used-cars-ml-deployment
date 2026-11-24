"""Standalone training pipeline for used car price category classification."""

from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "autos_enriched_dirty_english.csv"
MODEL_PATH = BASE_DIR / "models" / "autos_model.joblib"
INPUT_JSON_PATH = BASE_DIR / "input.json"

MODEL_VERSION = "1.0.0"
PRICE_THRESHOLD_EUR = 3500

SELECTED_COLUMNS = [
    "price_eur",
    "registration_year",
    "horsepower_ps",
    "mileage_km",
    "brand",
    "vehicle_type",
    "transmission",
    "fuel_type",
    "unrepaired_damage_reported",
]
NUMERIC_SOURCE_COLS = [
    "price_eur",
    "registration_year",
    "horsepower_ps",
    "mileage_km",
]
NUMERIC_FEATURE_COLS = [
    "registration_year",
    "horsepower_ps",
    "mileage_km",
]
NOMINAL_FEATURE_COLS = [
    "brand",
    "vehicle_type",
    "transmission",
    "fuel_type",
    "unrepaired_damage_reported",
]
TRANSLATIONS = {
    "vehicle_type": {
        "kleinwagen": "small_car",
        "limousine": "sedan",
        "kombi": "station_wagon",
        "bus": "van",
        "cabrio": "convertible",
        "andere": "other",
    },
    "transmission": {
        "manuell": "manual",
        "automatik": "automatic",
    },
    "fuel_type": {
        "benzin": "petrol",
    },
    "unrepaired_damage_reported": {
        "nein": "no",
        "ja": "yes",
    },
}


def load_raw_dataset(data_path: Path) -> pd.DataFrame:
    """Load the raw CSV dataset or fail clearly if it is missing."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Fichier de données introuvable : {data_path}. "
            "Placez 'autos_enriched_dirty_english.csv' dans le dossier 'data/'."
        )
    return pd.read_csv(
        data_path,
        sep=";",
        encoding="utf-8-sig",
        low_memory=False,
    )


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
    """Clean raw listings, harmonize categories, impute nulls, and remove outliers."""
    df = raw_df.drop_duplicates()
    df = df[SELECTED_COLUMNS].copy()

    for col in NUMERIC_SOURCE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col, mapping in TRANSLATIONS.items():
        df[col] = df[col].replace(mapping)

    df = df.dropna(
        subset=["price_eur", "registration_year", "horsepower_ps", "mileage_km", "brand"]
    )

    df = df[
        df["price_eur"].between(100, 150000)
        & df["registration_year"].between(1960, 2016)
        & df["horsepower_ps"].between(20, 1000)
    ].copy()

    for col in ("transmission", "fuel_type"):
        mode_series = df[col].mode(dropna=True)
        if mode_series.empty:
            raise ValueError(f"Impossible de calculer le mode pour la colonne '{col}'.")
        df[col] = df[col].fillna(mode_series.iloc[0])

    for col in ("vehicle_type", "unrepaired_damage_reported"):
        df[col] = df[col].fillna("unknown")

    df = remove_iqr_outliers(df, ["registration_year", "horsepower_ps"])

    for col in NUMERIC_FEATURE_COLS:
        df[col] = df[col].round().astype(int)

    return df


def build_pipeline() -> Pipeline:
    """Create the scikit-learn preprocessing and RandomForest classifier pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURE_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                NOMINAL_FEATURE_COLS,
            ),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def export_sample_inputs(x_test: pd.DataFrame, output_path: Path) -> None:
    """Export two valid sample records from X_test matching API input bounds."""
    valid_samples = x_test[
        x_test["registration_year"].between(1985, 2016)
        & x_test["horsepower_ps"].between(20, 255)
        & x_test["mileage_km"].between(5000, 150000)
    ]
    sample_df = valid_samples.head(2) if len(valid_samples) >= 2 else x_test.head(2)
    sample_df.to_json(output_path, orient="records", indent=2)


def train_and_save() -> None:
    """Run end-to-end training, evaluation, serialization, and sample export."""
    raw_df = load_raw_dataset(DATA_PATH)
    cleaned_df = clean_dataset(raw_df)

    price_category = cleaned_df["price_eur"].map(
        lambda price: "expensive" if price >= PRICE_THRESHOLD_EUR else "affordable"
    )
    features = cleaned_df.drop(columns=["price_eur"])

    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(price_category)

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
