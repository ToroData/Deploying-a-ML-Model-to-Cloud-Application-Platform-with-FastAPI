import os
import uvicorn
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ml.train_model import process_data

app = FastAPI()

class CensusData(BaseModel):
    age: int
    workclass: str 
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str
    
    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
            }
        }

# path to saved artifacts
savepath = './model'
filename = ['model.pkl', 'encoder.pkl', 'lb.pkl']

@app.get("/")
async def root():
    return {"message": "Welcome to the Census Income Prediction API!"}

@app.post("/inference/")
async def make_inference(data: CensusData):
    data = {  'age': data.age,
                'workclass': data.workclass, 
                'fnlgt': data.fnlgt,
                'education': data.education,
                'education-num': data.education_num,
                'marital-status': data.marital_status,
                'occupation': data.occupation,
                'relationship': data.relationship,
                'race': data.race,
                'sex': data.sex,
                'capital-gain': data.capital_gain,
                'capital-loss': data.capital_loss,
                'hours-per-week': data.hours_per_week,
                'native-country': data.native_country,
                }
    sample = pd.DataFrame(data, index=[0])

    # apply transformation to sample data
    cat_features = [
                    "workclass",
                    "education",
                    "marital-status",
                    "occupation",
                    "relationship",
                    "race",
                    "sex",
                    "native-country",
                    ]

    model = pickle.load(open(os.path.join(savepath,filename[0]), "rb"))
    encoder = pickle.load(open(os.path.join(savepath,filename[1]), "rb"))
    lb = pickle.load(open(os.path.join(savepath,filename[2]), "rb"))
        
    sample, _, _, _ = process_data(
                                sample, 
                                categorical_features=cat_features, 
                                training=False, 
                                encoder=encoder, 
                                lb=lb
                                )
                          
    prediction = model.predict(sample)

    if prediction[0]>0.5:
        prediction = '>50K'
    else:
        prediction = '<=50K', 
    data['prediction'] = prediction


    return data

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
