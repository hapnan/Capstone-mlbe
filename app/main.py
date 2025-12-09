from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

from app.utils import CustomImputer, CyclicalFeatureTransformer


# KONFIGURASI MODEL

MODEL_PATH = os.path.join("model", "model_lead_scoring_final_deployment.joblib")

# Load model saat startup
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model berhasil dimuat dari {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"Gagal memuat model: {e}")


app = FastAPI(
    title="Bank Lead Scoring API",
    description="API untuk memprediksi probabilitas nasabah melakukan deposit (Lead Score)."
)


# SKEMA INPUT SESUAI 19 FITUR TRAINING
class CustomerFeatures(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float

@app.get("/")
def home():
    return {"status": "API is running successfully"}

@app.post("/predict")
def predict_deposit(features: CustomerFeatures):

    data = features.model_dump()

    # Mapping ke nama kolom training (pakai titik)
    df_input = pd.DataFrame([{
        "age": data["age"],
        "job": data["job"],
        "marital": data["marital"],
        "education": data["education"],
        "default": data["default"],
        "housing": data["housing"],
        "loan": data["loan"],
        "contact": data["contact"],
        "month": data["month"],
        "day_of_week": data["day_of_week"],
        "campaign": data["campaign"],
        "pdays": data["pdays"],
        "previous": data["previous"],
        "poutcome": data["poutcome"],
        "emp.var.rate": data["emp_var_rate"],
        "cons.price.idx": data["cons_price_idx"],
        "cons.conf.idx": data["cons_conf_idx"],
        "euribor3m": data["euribor3m"],
        "nr.employed": data["nr_employed"]
    }])

    # Prediksi
    try:
        proba = float(model.predict_proba(df_input)[:, 1][0])
        pred = int(model.predict(df_input)[0])
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

    return {
        "prediction_label": "Deposit" if pred == 1 else "No Deposit",
        "lead_score_probability": round(proba, 4),
        "status": "success"
    }
