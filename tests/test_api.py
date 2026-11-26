"""Automated FastAPI integration and schema validation tests."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

EXPENSIVE_CAR = {
    "registration_year": 2025,
    "mileage_km": 15000,
    "horsepower_ps": 180,
    "engine_size_l": 2.0,
    "previous_owners": 1,
    "brand": "bmw",
    "vehicle_type": "suv",
    "fuel_type": "hybrid",
    "transmission": "automatic",
    "unrepaired_damage": "no",
}

CHEAP_CAR = {
    "registration_year": 2012,
    "mileage_km": 210000,
    "horsepower_ps": 70,
    "engine_size_l": 1.2,
    "previous_owners": 4,
    "brand": "dacia",
    "vehicle_type": "hatchback",
    "fuel_type": "petrol",
    "transmission": "manual",
    "unrepaired_damage": "yes",
}

AFFORDABLE_CAR = {
    "registration_year": 2021,
    "mileage_km": 75000,
    "horsepower_ps": 115,
    "engine_size_l": 1.6,
    "previous_owners": 2,
    "brand": "toyota",
    "vehicle_type": "sedan",
    "fuel_type": "petrol",
    "transmission": "manual",
    "unrepaired_damage": "no",
}


@pytest.fixture(scope="module")
def client():
    """Provide a TestClient initialized with FastAPI lifespan events."""
    with TestClient(app) as c:
        yield c


def test_root(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "API Classification Prix Véhicules 2026 opérationnelle"
    }


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert isinstance(payload["model_version"], str)


def test_predict_expensive_car(client: TestClient) -> None:
    response = client.post("/predict", json=EXPENSIVE_CAR)
    assert response.status_code == 200
    assert response.json()["class_name"] == "expensive"


def test_predict_cheap_car(client: TestClient) -> None:
    response = client.post("/predict", json=CHEAP_CAR)
    assert response.status_code == 200
    assert response.json()["class_name"] == "cheap"


def test_predict_affordable_car(client: TestClient) -> None:
    response = client.post("/predict", json=AFFORDABLE_CAR)
    assert response.status_code == 200
    assert response.json()["class_name"] == "affordable"


def test_predict_missing_field(client: TestClient) -> None:
    incomplete = dict(EXPENSIVE_CAR)
    del incomplete["brand"]
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_out_of_range(client: TestClient) -> None:
    invalid = dict(EXPENSIVE_CAR, registration_year=2050)
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_predict_unknown_category(client: TestClient) -> None:
    invalid = dict(EXPENSIVE_CAR, brand="ferrari")
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_predict_extra_field_forbidden(client: TestClient) -> None:
    invalid = dict(EXPENSIVE_CAR, color="red")
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_probabilities_sum_to_one(client: TestClient) -> None:
    response = client.post("/predict", json=EXPENSIVE_CAR)
    assert response.status_code == 200
    probs = response.json()["probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 1e-6
