"""Pydantic v2 request and response schemas for the 2026 used cars price API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CarFeatures(BaseModel):
    """Input vehicle features validated strictly against the 2026 training domain."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    registration_year: int = Field(
        ...,
        ge=2010,
        le=2026,
        description="Année de mise en circulation",
        examples=[2022],
    )
    mileage_km: int = Field(
        ...,
        ge=0,
        le=300000,
        description="Kilométrage (km)",
        examples=[65000],
    )
    horsepower_ps: int = Field(
        ...,
        ge=50,
        le=300,
        description="Puissance en chevaux (PS)",
        examples=[130],
    )
    engine_size_l: float = Field(
        ...,
        ge=0.0,
        le=4.0,
        description="Cylindrée (L) - 0.0 pour électrique",
        examples=[1.6],
    )
    previous_owners: int = Field(
        ...,
        ge=1,
        le=5,
        description="Nombre de propriétaires précédents",
        examples=[1],
    )
    brand: Literal[
        "audi",
        "bmw",
        "citroen",
        "dacia",
        "fiat",
        "ford",
        "hyundai",
        "kia",
        "mercedes_benz",
        "opel",
        "peugeot",
        "renault",
        "seat",
        "skoda",
        "tesla",
        "toyota",
        "volkswagen",
        "volvo",
    ] = Field(..., examples=["toyota"])
    vehicle_type: Literal[
        "convertible",
        "coupe",
        "hatchback",
        "sedan",
        "station_wagon",
        "suv",
        "unknown",
        "van",
    ] = Field(..., examples=["suv"])
    fuel_type: Literal[
        "diesel",
        "electric",
        "hybrid",
        "lpg",
        "petrol",
    ] = Field(..., examples=["hybrid"])
    transmission: Literal["automatic", "manual"] = Field(..., examples=["automatic"])
    unrepaired_damage: Literal["no", "unknown", "yes"] = Field(..., examples=["no"])


class PredictionResponse(BaseModel):
    """Structured 3-class prediction response returned by the inference endpoint."""

    class_id: int
    class_name: str
    probabilities: dict[str, float]
    model_version: str
