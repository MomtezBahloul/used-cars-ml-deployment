"""Standalone Streamlit UI for estimating used vehicle price categories."""

from pathlib import Path
import sys

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.model import load_model, predict
from app.schemas import CarFeatures

BRANDS = [
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
]

VEHICLE_TYPES = [
    "convertible",
    "coupe",
    "other",
    "sedan",
    "small_car",
    "station_wagon",
    "suv",
    "unknown",
    "van",
]

TRANSMISSIONS = ["automatic", "manual"]

FUEL_TYPES = [
    "cng",
    "diesel",
    "electric",
    "hybrid",
    "lpg",
    "other",
    "petrol",
]

DAMAGE_OPTIONS = ["no", "unknown", "yes"]


@st.cache_resource
def get_model_bundle():
    """Load and cache the trained scikit-learn pipeline bundle in memory."""
    return load_model()


st.set_page_config(
    page_title="Estimation Catégorie de Prix Véhicule",
    layout="centered",
)
st.title("Estimation Catégorie de Prix Véhicule")

left_col, right_col = st.columns(2)

with left_col:
    registration_year = st.number_input(
        "Année de mise en circulation",
        min_value=1985,
        max_value=2016,
        value=2012,
        step=1,
    )
    horsepower_ps = st.number_input(
        "Puissance (PS)",
        min_value=20,
        max_value=255,
        value=140,
        step=1,
    )
    mileage_km = st.number_input(
        "Kilométrage (km)",
        min_value=5000,
        max_value=150000,
        value=80000,
        step=1000,
    )
    brand = st.selectbox("Marque", BRANDS, index=BRANDS.index("audi"))

with right_col:
    vehicle_type = st.selectbox(
        "Type de véhicule",
        VEHICLE_TYPES,
        index=VEHICLE_TYPES.index("sedan"),
    )
    transmission = st.selectbox(
        "Transmission",
        TRANSMISSIONS,
        index=TRANSMISSIONS.index("automatic"),
    )
    fuel_type = st.selectbox(
        "Carburant",
        FUEL_TYPES,
        index=FUEL_TYPES.index("diesel"),
    )
    unrepaired_damage_reported = st.selectbox(
        "Dommages non réparés",
        DAMAGE_OPTIONS,
        index=DAMAGE_OPTIONS.index("no"),
    )

if st.button("Prédire"):
    payload = {
        "registration_year": int(registration_year),
        "horsepower_ps": int(horsepower_ps),
        "mileage_km": int(mileage_km),
        "brand": brand,
        "vehicle_type": vehicle_type,
        "transmission": transmission,
        "fuel_type": fuel_type,
        "unrepaired_damage_reported": unrepaired_damage_reported,
    }
    try:
        validated = CarFeatures(**payload)
        bundle = get_model_bundle()
        result = predict(bundle, validated.model_dump())
    except FileNotFoundError as exc:
        st.error(f"Modèle introuvable : {exc}")
    except Exception as exc:
        st.error(f"Erreur de prédiction : {exc}")
    else:
        st.success(f"Catégorie prédite : {result['class_name']}")
        st.caption(f"Version du modèle : {result['model_version']}")
        st.bar_chart(result["probabilities"])
