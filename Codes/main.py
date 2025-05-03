from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import json

app = FastAPI()

# Load the XGBoost model
model = xgb.Booster()
model.load_model("xgboost_model.json")

# Define request body
class InputData(BaseModel):
    name: str
    minutes: float
    ingredients: str
    preparation_method: str
    description: str
    calories: float
    fat: float
    sugar: float
    protein: float
    carb: float

@app.post("/predict")
def predict(data: InputData):
    # Convert the input data to the format your model expects (you may need to adjust this part)
    input_array = np.array([
        [data.minutes, data.calories, data.fat, data.sugar, data.protein, data.carb]
    ])
    
    # Prediction
    dmatrix = xgb.DMatrix(input_array)
    prediction = model.predict(dmatrix)
    
    return {"prediction": prediction.tolist()}
