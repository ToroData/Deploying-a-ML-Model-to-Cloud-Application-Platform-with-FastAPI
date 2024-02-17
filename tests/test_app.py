"""Script to test the app.
Author: Ricard Santiago Raigada García
Date: February, 2024
"""
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_read_main():
    """
    Test para el endpoint GET en la raíz.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the Census Income Prediction API!"}


def test_post_inference_less_than_50k():
    """
    Test para el endpoint POST que debería predecir ingresos <=50K.
    """
    data = {
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
    }  # line 1 census.csv
    response = client.post("/inference/", json=data)
    assert response.status_code == 200
    assert response.json()["age"] == 39
    assert response.json()["prediction"][0] == "<=50K"


def test_post_inference_greater_than_50k():
    """
    Test para el endpoint POST que debería predecir ingresos >50K.
    """
    data = {'age': 30,
            'workclass': "Private",
            'fnlgt': 234721,
            'education': "HS-grad",
            'education_num': 1,
            'marital_status': "Separated",
            'occupation': "Handlers-cleaners",
            'relationship': "Not-in-family",
            'race': "Black",
            'sex': "Male",
            'capital_gain': 0,
            'capital_loss': 0,
            'hours_per_week': 35,
            'native_country': "United-States"
            }  # line 9 census.csv
    response = client.post("/inference/", json=data)
    assert response.status_code == 200
    assert response.json()["age"] == 30
