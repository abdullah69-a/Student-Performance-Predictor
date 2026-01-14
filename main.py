from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pickle

app=FastAPI()

@app.get("/")
def read():
    return {"Message":"The fastapi is working ."}

model=joblib.load(open('student_model.pkl','rb'))
scaler=joblib.load(open('scaler.pkl','rb'))
class StudentData(BaseModel):
    hours_studied: float
    previous_scores: float
    extracurricular: int 
    sleep_hours: float
    papers_practiced: float

@app.post('/prediction')
def predict(data:StudentData):
    features = np.array([[data.hours_studied, data.previous_scores, 
                          data.extracurricular, data.sleep_hours, 
                          data.papers_practiced]])
    features_scaled=scaler.transform(features)
    Prediction=model.predict(features_scaled)

    return {"Performance_Index": round(Prediction[0], 2)}
