import requests

url = 'https://udacity-hvg7.onrender.com/inference/'

data = {'age': 50,
        'workclass': "Private",
        'fnlgt': 234721,
        'education': "Doctorate",
        'education_num': 16,
        'marital_status': "Separated",
        'occupation': "Exec-managerial",
        'relationship': "Not-in-family",
        'race': "Black",
        'sex': "Female",
        'capital_gain': 0,
        'capital_loss': 0,
        'hours_per_week': 50,
        'native_country': "United-States"
        }

# POST
response = requests.post(url, json=data)

if response.status_code == 200:
    print("Successfull.")
    print(response.json())
else:
    print(f"Error. Code: {response.status_code}")
    print(response.text)
