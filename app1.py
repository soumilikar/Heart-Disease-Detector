from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Annotated
import pickle
import pandas as pd

# Import the ML model
with open("model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
model_columns = model_data["columns"]

app = FastAPI()

class UserInput(BaseModel):
    age: Annotated[int, Field(..., alias="Age", description="Age of the patient in years")]
    sex: Annotated[str, Field(..., alias="Sex", description="Sex of the patient [M, F]")]
    chest_pain_type: Annotated[str, Field(..., alias="ChestPainType", description="Chest pain type [TA, ATA, NAP, ASY]")]
    resting_bp: Annotated[int, Field(..., alias="RestingBP", description="Resting blood pressure [mm Hg]")]
    cholesterol: Annotated[int, Field(..., alias="Cholesterol", description="Serum cholesterol [mm/dl]")]
    fasting_bs: Annotated[int, Field(..., alias="FastingBS", description="Fasting blood sugar [1, 0]")]
    resting_ecg: Annotated[str, Field(..., alias="RestingECG", description="Resting ECG results [Normal, ST, LVH]")]
    max_hr: Annotated[int, Field(..., alias="MaxHR", description="Maximum heart rate achieved")]
    exercise_angina: Annotated[str, Field(..., alias="ExerciseAngina", description="Exercise-induced angina [Y, N]")]
    oldpeak: Annotated[float, Field(..., alias="Oldpeak", description="ST depression measured in depression")]
    st_slope: Annotated[str, Field(..., alias="ST_Slope", description="Slope of peak exercise ST segment [Up, Flat, Down]")]

#human readable 
@app.get('/')
def home():
    return {'message':'Heart Disease Prediction API'}

#to check if api is live and working properly (for Kubernetes and AWS Cloud services)
@app.get('/health')
def health_check():
    return {
        'status': 'OK',
        'model_loaded': model is not None
    }

@app.post('/predict')
def predict_heart_disease(data: UserInput):
    input_df = pd.DataFrame([data.model_dump(by_alias=True)])

    # Convert categorical variables to dummy variables
    input_df = pd.get_dummies(
        input_df,
        columns=[
            "Sex",
            "ChestPainType",
            "RestingECG",
            "ExerciseAngina",
            "ST_Slope"
        ]
    )

    # Make sure input has exactly the same columns as training data
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    prediction = int(model.predict(input_df)[0])

    return {
        "prediction": prediction,
        "status": "Heart Disease" if prediction == 1 else "Normal"
    }