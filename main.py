from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import joblib
import os

app = FastAPI(
    title="Iris Species Classifier",
    description="A Random Forest ML model trained on the Iris dataset.",
    version="1.0.0",
)

MODEL_PATH = "iris_model.pkl"


def train_and_save_model():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    print(f"Model trained — accuracy: {accuracy:.4f}")
    joblib.dump(
        {
            "model": clf,
            "feature_names": list(iris.feature_names),
            "target_names": list(iris.target_names),
        },
        MODEL_PATH,
    )
    return clf, list(iris.feature_names), list(iris.target_names)


# Load or train on startup
if os.path.exists(MODEL_PATH):
    _saved = joblib.load(MODEL_PATH)
    _model: RandomForestClassifier = _saved["model"]
    _feature_names: list[str] = _saved["feature_names"]
    _target_names: list[str] = _saved["target_names"]
    print("Model loaded from disk.")
else:
    _model, _feature_names, _target_names = train_and_save_model()


# ── Schemas ──────────────────────────────────────────────────────────────────

class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., example=5.1, description="Sepal length in cm")
    sepal_width: float  = Field(..., example=3.5, description="Sepal width in cm")
    petal_length: float = Field(..., example=1.4, description="Petal length in cm")
    petal_width: float  = Field(..., example=0.2, description="Petal width in cm")


class PredictionResponse(BaseModel):
    species: str
    prediction: int
    probabilities: dict[str, float]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "message": "Iris Classifier API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health", tags=["Info"])
def health():
    return {
        "status": "ok",
        "model": "RandomForestClassifier",
        "classes": _target_names,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["ML"])
def predict(features: IrisFeatures):
    X = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]])
    try:
        pred = int(_model.predict(X)[0])
        probs = _model.predict_proba(X)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictionResponse(
        species=_target_names[pred],
        prediction=pred,
        probabilities={
            name: round(float(p), 4)
            for name, p in zip(_target_names, probs)
        },
    )
