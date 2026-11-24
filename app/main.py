"""FastAPI application entrypoint for vehicle price category classification."""

from contextlib import asynccontextmanager
import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException

from app.model import MODEL_PATH, load_model, predict
from app.schemas import CarFeatures, PredictionResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("autos-ml-api")

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model bundle once on startup and release state on shutdown."""
    try:
        state["model"] = load_model()
        logger.info(
            "Model loaded successfully version=%s path=%s",
            state["model"].get("version"),
            MODEL_PATH,
        )
    except FileNotFoundError:
        logger.error("Model file missing at startup: %s", MODEL_PATH)
    except Exception:
        logger.exception("Failed to load model at startup")
    try:
        yield
    finally:
        state.clear()


app = FastAPI(
    title="API Classification Prix Véhicules",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def read_root() -> dict[str, str]:
    """Return a basic operational status message."""
    return {"message": "API Classification Prix Véhicules opérationnelle"}


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Report service health and whether the model bundle is loaded."""
    bundle = state.get("model")
    return {
        "status": "ok",
        "model_loaded": bundle is not None,
        "model_version": bundle["version"] if bundle is not None else None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_price_category(features: CarFeatures) -> dict[str, Any]:
    """Predict whether a used car belongs to the affordable or expensive category."""
    bundle = state.get("model")
    if bundle is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    start = time.perf_counter()
    try:
        result = predict(bundle, features.model_dump())
    except Exception as exc:
        logger.exception("Erreur lors de la prédiction")
        raise HTTPException(
            status_code=500,
            detail="Erreur interne de prédiction",
        ) from exc

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    logger.info(
        "predict ms=%.1f version=%s class=%s",
        elapsed_ms,
        result["model_version"],
        result["class_name"],
    )
    return result
