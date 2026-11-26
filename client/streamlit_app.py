"""Standalone Streamlit UI for 3-class used car price classification (2026)."""

from pathlib import Path
import sys

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.model import load_model, predict
from app.schemas import CarFeatures

BRANDS = [
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
]

VEHICLE_TYPES = [
    "convertible",
    "coupe",
    "hatchback",
    "sedan",
    "station_wagon",
    "suv",
    "unknown",
    "van",
]

FUEL_TYPES = [
    "diesel",
    "electric",
    "hybrid",
    "lpg",
    "petrol",
]

TRANSMISSIONS = ["automatic", "manual"]

DAMAGE_OPTIONS = ["no", "unknown", "yes"]


@st.cache_resource
def get_model_bundle():
    """Load and cache the trained scikit-learn pipeline bundle in memory."""
    return load_model()


st.set_page_config(
    page_title="Estimation Catégorie de Prix Véhicule (2026)",
    layout="centered",
)
st.title("Estimation Catégorie de Prix Véhicule (2026)")

left_col, right_col = st.columns(2)

with left_col:
    registration_year = st.number_input(
        "Année de mise en circulation",
        min_value=2010,
        max_value=2026,
        value=2022,
        step=1,
    )
    mileage_km = st.number_input(
        "Kilométrage (km)",
        min_value=0,
        max_value=300000,
        value=65000,
        step=1000,
    )
    horsepower_ps = st.number_input(
        "Puissance (PS)",
        min_value=50,
        max_value=300,
        value=130,
        step=1,
    )
    engine_size_l = st.number_input(
        "Cylindrée (L) - 0.0 pour électrique",
        min_value=0.0,
        max_value=4.0,
        value=1.6,
        step=0.1,
    )
    previous_owners = st.number_input(
        "Nombre de propriétaires précédents",
        min_value=1,
        max_value=5,
        value=1,
        step=1,
    )

with right_col:
    brand = st.selectbox("Marque", BRANDS, index=BRANDS.index("toyota"))
    vehicle_type = st.selectbox(
        "Type de véhicule",
        VEHICLE_TYPES,
        index=VEHICLE_TYPES.index("suv"),
    )
    fuel_type = st.selectbox(
        "Carburant",
        FUEL_TYPES,
        index=FUEL_TYPES.index("hybrid"),
    )
    transmission = st.selectbox(
        "Transmission",
        TRANSMISSIONS,
        index=TRANSMISSIONS.index("automatic"),
    )
    unrepaired_damage = st.selectbox(
        "Dommages non réparés",
        DAMAGE_OPTIONS,
        index=DAMAGE_OPTIONS.index("no"),
    )

if st.button("Prédire"):
    payload = {
        "registration_year": int(registration_year),
        "mileage_km": int(mileage_km),
        "horsepower_ps": int(horsepower_ps),
        "engine_size_l": round(float(engine_size_l), 1),
        "previous_owners": int(previous_owners),
        "brand": brand,
        "vehicle_type": vehicle_type,
        "fuel_type": fuel_type,
        "transmission": transmission,
        "unrepaired_damage": unrepaired_damage,
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
