"""Automated FastAPI integration and schema validation tests."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

EXPENSIVE_CAR = {
    "registration_year": 2014,
    "horsepower_ps": 190,
    "mileage_km": 40000,
    "brand": "audi",
    "vehicle_type": "suv",
    "transmission": "automatic",
    "fuel_type": "diesel",
    "unrepaired_damage_reported": "no",
}

AFFORDABLE_CAR = {
    "registration_year": 1998,
    "horsepower_ps": 60,
    "mileage_km": 150000,
    "brand": "opel",
    "vehicle_type": "small_car",
    "transmission": "manual",
    "fuel_type": "petrol",
    "unrepaired_damage_reported": "yes",
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
        "message": "API Classification Prix Véhicules opérationnelle"
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
    payload = response.json()
    assert payload["class_name"] == "expensive"


def test_predict_affordable_car(client: TestClient) -> None:
    response = client.post("/predict", json=AFFORDABLE_CAR)
    assert response.status_code == 200
    payload = response.json()
    assert payload["class_name"] == "affordable"


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
