from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.predict_wrapper import ReadmissionPredictor

app = FastAPI(root_path="/proxy/8000") 
model = ReadmissionPredictor(model_uri="models:/xgb-readmission/8")  # 🔁 adapte la version si besoin

# ✅ Schéma des entrées attendues
class InputData(BaseModel):
    chol: float
    crp: float
    phos: float

# 🏠 Page d'accueil
@app.get("/")
def root():
    return {"message": "API is ready!"}

# 🔮 Prédiction
@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}
