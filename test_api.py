"""
test_api.py
Simple client that sends requests to the running API.

1. Start the server first:  fastapi dev main.py
2. Then run:                python test_api.py
"""

import requests

BASE_URL = "http://localhost:8000"


def show(title, response):
    print(f"--- {title} ---")
    print("Status code:", response.status_code)
    print("Response:", response.json())
    print()


# GET /
show("GET /", requests.get(f"{BASE_URL}/"))

# GET /health
show("GET /health", requests.get(f"{BASE_URL}/health"))

# POST /predict - three sample flowers (one of each species)
samples = {
    "Expected setosa": {
        "sepal_length": 5.1, "sepal_width": 3.5,
        "petal_length": 1.4, "petal_width": 0.2,
    },
    "Expected versicolor": {
        "sepal_length": 6.0, "sepal_width": 2.9,
        "petal_length": 4.5, "petal_width": 1.5,
    },
    "Expected virginica": {
        "sepal_length": 6.9, "sepal_width": 3.1,
        "petal_length": 5.4, "petal_width": 2.1,
    },
}

for label, data in samples.items():
    show(f"POST /predict ({label})", requests.post(f"{BASE_URL}/predict", json=data))

# Invalid input example (missing field) -> FastAPI returns 422
show("POST /predict (invalid input)",
     requests.post(f"{BASE_URL}/predict", json={"sepal_length": 5.1}))
