# Autos ML Deployment

Production-ready machine learning service built with Python 3.12, Scikit-Learn, FastAPI, Pydantic v2, pytest, and Streamlit for binary used-car price classification (`affordable` vs `expensive`, threshold 3,500 EUR).

## Project Structure

```text
autos-ml-deployment/
├── app/
│   ├── __init__.py
│   ├── schemas.py
│   ├── model.py
│   └── main.py
├── training/
│   └── train_model.py
├── client/
│   └── streamlit_app.py
├── tests/
│   └── test_api.py
├── data/
│   └── autos_enriched_dirty_english.csv
├── models/
│   └── autos_model.joblib
├── input.json
├── requirements.txt
├── .python-version
├── pytest.ini
├── .gitignore
└── README.md
```

## Setup & Usage (Windows PowerShell)

All commands below assume your current directory is `autos-ml-deployment`.

### 1. Create and activate the Python 3.12 virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Prepare data and train the model

Ensure `autos_enriched_dirty_english.csv` is placed inside `data/`:

```powershell
Copy-Item ..\autos_enriched_dirty_english.csv .\data\autos_enriched_dirty_english.csv
python training\train_model.py
```

This generates `models\autos_model.joblib` and sample records in `input.json`.

### 4. Run the FastAPI server (optional REST API)

```powershell
uvicorn app.main:app --reload
```

Interactive Swagger docs will be available at `http://127.0.0.1:8000/docs`.

### 5. Run the automated test suite

```powershell
pytest -v
```

### 6. Launch the Streamlit client

`client\streamlit_app.py` runs standalone by loading `models\autos_model.joblib` directly in-process (no need to run FastAPI alongside it):

```powershell
streamlit run client\streamlit_app.py
```

## Free Cloud Deployment (Streamlit Community Cloud)

1. **Push to GitHub:** Ensure `models/autos_model.joblib` is committed (`data/*.csv` and `.venv/` remain ignored via `.gitignore`).
2. **Deploy on Streamlit Community Cloud (`https://share.streamlit.io`):**
   - **Repository:** `Momtez-bahloul/autos-ml-deployment`
   - **Branch:** `main`
   - **Main file path:** `client/streamlit_app.py`
   - **Advanced settings -> Python version:** `3.12`
   - No environment variables or secrets are required.
