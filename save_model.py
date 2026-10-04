"""
save_model.py
Trains an Iris flower classifier and saves the full pipeline
(StandardScaler + LogisticRegression) to model/pipeline.pkl.

Run:  python save_model.py
"""

import os

import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Feature names used by the API (must match the Pydantic schema in main.py)
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def main():
    # 1. Load data
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=FEATURES)
    # Store species names (setosa / versicolor / virginica) as the target labels
    y = pd.Series(iris.target_names[iris.target], name="species")

    # 2. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Build pipeline: preprocessing + model saved together
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])

    # 4. Train
    pipeline.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = pipeline.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred))

    # 6. Save
    os.makedirs("model", exist_ok=True)
    joblib.dump(pipeline, "model/pipeline.pkl")
    print("Pipeline saved to model/pipeline.pkl")


if __name__ == "__main__":
    main()
