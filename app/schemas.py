"""Pydantic v2 request and response schemas for the vehicle price API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CarFeatures(BaseModel):
    """Input vehicle features validated strictly against the training domain."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    registration_year: int = Field(
        ...,
        ge=1985,
        le=2016,
        description="Année de mise en circulation",
        examples=[2012],
    )
    horsepower_ps: int = Field(
        ...,
        ge=20,
        le=255,
        description="Puissance en chevaux (PS)",
        examples=[140],
    )
    mileage_km: int = Field(
        ...,
        ge=5000,
        le=150000,
        description="Kilométrage (km)",
        examples=[80000],
    )
    brand: Literal[
        "alfa_romeo",
        "audi",
        "bmw",
        "chevrolet",
        "chrysler",
        "citroen",
        "dacia",
        "daewoo",
        "daihatsu",
        "fiat",
        "ford",
        "honda",
        "hyundai",
        "jaguar",
        "jeep",
        "kia",
        "lada",
        "lancia",
        "land_rover",
        "mazda",
        "mercedes_benz",
        "mini",
        "mitsubishi",
        "nissan",
        "opel",
        "peugeot",
        "porsche",
        "renault",
        "rover",
        "saab",
        "seat",
        "skoda",
        "smart",
        "sonstige_autos",
        "subaru",
        "suzuki",
        "toyota",
        "trabant",
        "volkswagen",
        "volvo",
    ] = Field(..., examples=["audi"])
    vehicle_type: Literal[
        "convertible",
        "coupe",
        "other",
        "sedan",
        "small_car",
        "station_wagon",
        "suv",
        "unknown",
        "van",
    ] = Field(..., examples=["sedan"])
    transmission: Literal["automatic", "manual"] = Field(..., examples=["automatic"])
    fuel_type: Literal[
        "cng",
        "diesel",
        "electric",
        "hybrid",
        "lpg",
        "other",
        "petrol",
    ] = Field(..., examples=["diesel"])
    unrepaired_damage_reported: Literal["no", "unknown", "yes"] = Field(
        ...,
        examples=["no"],
    )


class PredictionResponse(BaseModel):
    """Structured prediction response returned by the inference endpoint."""

    class_id: int
    class_name: str
    probabilities: dict[str, float]
    model_version: str
