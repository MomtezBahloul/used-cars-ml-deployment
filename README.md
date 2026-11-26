# Used Cars ML Deployment (2026)

Production-ready machine learning service built with Python 3.12, Scikit-Learn, FastAPI, Pydantic v2, pytest, and Streamlit for 3-class used car price classification (`cheap`, `affordable`, `expensive`).

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
│   └── used_cars_2026_dirty.csv
├── models/
│   └── used_cars_model.joblib
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

Ensure `used_cars_2026_dirty.csv` is placed inside `data\`, then run:

```powershell
python training\train_model.py
```

This generates `models\used_cars_model.joblib` and 2 sample records in `input.json`.

### 4. Run the FastAPI server (optional REST API)

```powershell
uvicorn app.main:app --reload
```

Interactive Swagger docs will be available at `http://127.0.0.1:8000/docs`.

### 5. Run the automated test suite

```powershell
pytest -v
```

### 6. Launch the standalone Streamlit client

`client\streamlit_app.py` loads `models\used_cars_model.joblib` directly in-process (no separate FastAPI server required):

```powershell
streamlit run client\streamlit_app.py
```

## Free Cloud Deployment (Streamlit Community Cloud)

1. **GitHub repository:** `MomtezBahloul/used-cars-ml-deployment` (branch `main`, with `models/used_cars_model.joblib` tracked and `data/*.csv` / `.venv/` ignored via `.gitignore`).
2. **Streamlit Community Cloud (`https://share.streamlit.io`):**
   - **Main file path:** `client/streamlit_app.py`
   - **Python version:** `3.12`
   - Pushing to `main` automatically redeploys your Streamlit Community Cloud app.
