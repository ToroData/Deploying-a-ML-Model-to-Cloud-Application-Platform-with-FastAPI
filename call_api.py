import requests

# La URL del endpoint de tu API para realizar inferencias
url = 'http://127.0.0.1:8000/inference/'

# Reemplaza esto con los datos que coincidan con la estructura esperada por tu API
data = { 'age':50,
            'workclass':"Private", 
            'fnlgt':234721,
            'education':"Doctorate",
            'education_num':16,
            'marital_status':"Separated",
            'occupation':"Exec-managerial",
            'relationship':"Not-in-family",
            'race':"Black",
            'sex':"Female",
            'capital_gain':0,
            'capital_loss':0,
            'hours_per_week':50,
            'native_country':"United-States"
            }

# Realiza la solicitud POST
response = requests.post(url, json=data)

# Verifica que la solicitud fue exitosa
if response.status_code == 200:
    print("Solicitud exitosa.")
    # Imprime la respuesta de la inferencia
    print(response.json())
else:
    print(f"Error en la solicitud. Código de estado: {response.status_code}")
    print(response.text)