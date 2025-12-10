from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, List
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

class CustomerData(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Customer age")
    job: str = Field(..., description="Job type (e.g., admin., blue-collar, entrepreneur, etc.)")
    marital: str = Field(..., description="Marital status (married, single, divorced)")
    education: str = Field(..., description="Education level (e.g., basic.4y, high.school, university.degree)")
    default: str = Field(..., description="Has credit in default? (yes, no, unknown)")
    housing: str = Field(..., description="Has housing loan? (yes, no, unknown)")
    loan: str = Field(..., description="Has personal loan? (yes, no, unknown)")
    contact: str = Field(..., description="Contact communication type (cellular, telephone)")
    month: str = Field(..., description="Last contact month (jan, feb, mar, etc.)")
    day_of_week: str = Field(..., description="Last contact day of week (mon, tue, wed, thu, fri)")
    duration: int = Field(..., ge=0, description="Last contact duration in seconds")
    campaign: int = Field(..., ge=1, description="Number of contacts during this campaign")
    pdays: int = Field(..., ge=0, description="Days since last contact (999 means never contacted)")
    previous: int = Field(..., ge=0, description="Number of contacts before this campaign")
    poutcome: str = Field(..., description="Outcome of previous campaign (failure, nonexistent, success)")
    emp_var_rate: float = Field(..., description="Employment variation rate")
    cons_price_idx: float = Field(..., description="Consumer price index")
    cons_conf_idx: float = Field(..., description="Consumer confidence index")
    euribor3m: float = Field(..., description="Euribor 3 month rate")
    nr_employed: float = Field(..., description="Number of employees")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 56,
                "job": "housemaid",
                "marital": "married",
                "education": "basic.4y",
                "default": "no",
                "housing": "no",
                "loan": "no",
                "contact": "telephone",
                "month": "may",
                "day_of_week": "mon",
                "duration": 261,
                "campaign": 1,
                "pdays": 999,
                "previous": 0,
                "poutcome": "nonexistent",
                "emp_var_rate": 1.1,
                "cons_price_idx": 93.994,
                "cons_conf_idx": -36.4,
                "euribor3m": 4.857,
                "nr_employed": 5191.0
            }
        }
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

class PredictionRequest(BaseModel):
    data: CustomerFeatures | List[CustomerFeatures]


class SinglePrediction(BaseModel):
    prediction: int
    prediction_label: str
    probability: float


class PredictionResponse(BaseModel):
    predictions: List[SinglePrediction]
    count: int
    status: str

@app.get("/")
def home():
    return {"status": "API is running successfully"}

@app.post("/predict")
def predict_deposit(request: PredictionRequest):
    
    # Check if input is single or list
    customers = request.data if isinstance(request.data, list) else [request.data]
    
    # Convert all features to dataframe rows
    customers_list = []
    for customer in customers:
        customer_dict = customer.model_dump()
        # Rename keys to match expected column names
        customer_dict['emp.var.rate'] = customer_dict.pop('emp_var_rate')
        customer_dict['cons.price.idx'] = customer_dict.pop('cons_price_idx')
        customer_dict['cons.conf.idx'] = customer_dict.pop('cons_conf_idx')
        customer_dict['nr.employed'] = customer_dict.pop('nr_employed')
        customers_list.append(customer_dict)
    
    df_input = pd.DataFrame(customers_list)

    # Prediksi
    try:
        probas = model.predict_proba(df_input)[:, 1]
        preds = model.predict(df_input)
        
        # Build results
        results = []
        for i in range(len(df_input)):
            results.append({
                "prediction_label": "Deposit" if preds[i] == 1 else "No Deposit",
                "lead_score_probability": round(float(probas[i]), 4)
            })
        
        return PredictionResponse(
            predictions=results,
            count=len(results),
            status="success"
        )
            
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}
