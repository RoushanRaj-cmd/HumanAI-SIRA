import requests
import numpy as np

t = list(np.linspace(0, 10, 11))
S = [0.99, 0.95, 0.90, 0.82, 0.70, 0.55, 0.40, 0.28, 0.20, 0.15, 0.12]
I = [0.01, 0.05, 0.10, 0.18, 0.30, 0.45, 0.50, 0.62, 0.70, 0.75, 0.78]
R = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.10, 0.10, 0.10, 0.10] 

payload = {
    "t": t,
    "S": S,
    "I": I,
    "R": R,
    "epochs_adam": 500, 
    "epochs_lbfgs": 10
}

url = "http://127.0.0.1:8000/infer_parameters"
print(f"Sending POST request to {url} ...")

try:
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        print("\n Success! FastAPI Neural Network Extracted:")
        print(response.json())
    else:
        print(f"\n Failed with Status {response.status_code}:")
        print(response.text)
except requests.exceptions.ConnectionError:
    print("\n Connection Failed! Make sure you are running 'fastapi dev api/main.py' in another terminal tab first.")
