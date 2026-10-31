import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

COLUMNS = ["Sex", "Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight",
           "Rings"]
TARGET = "Rings"
NUMERIC_COLS = ["Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight"]
CATEGORICAL_COLS = ["Sex"]


def _app_base():
    if getattr(sys, "frozen", False): return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_PATH = os.path.join(_app_base(), "datasets", "abalone.data.csv")
MODEL_PATH = os.path.join(os.getcwd(), "models", "abalone_regression_model.pkl")


def load_data():
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)
    return df


def _build_pipeline(model):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False), CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERIC_COLS),
        ]
    )
    return Pipeline([("prep", preprocessor), ("model", model)])


def train_and_evaluate():
    df = load_data()
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
        "Ridge Regression": Ridge(alpha=1.0),
        "Linear Regression": LinearRegression(),
        "Lasso Regression": Lasso(alpha=0.01, max_iter=5000)
    }

    results = {}
    best_r2 = -float('inf')
    best_name = ""
    best_pipe = None

    for name, model in models.items():
        pipe = _build_pipeline(model)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        r2 = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))

        results[name] = {"r2": r2, "rmse": rmse, "mae": mae}

        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_pipe = pipe

    final_model = _build_pipeline(models[best_name])
    final_model.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": final_model, "metrics": results, "best_name": best_name}, MODEL_PATH)
    return results, best_name


def predict(sex, length, diameter, height, whole_weight, shucked_weight, viscera_weight, shell_weight):
    if not os.path.exists(MODEL_PATH): train_and_evaluate()
    bundle = joblib.load(MODEL_PATH)
    X_new = pd.DataFrame([{
        "Sex": sex, "Length": length, "Diameter": diameter, "Height": height,
        "Whole_weight": whole_weight, "Shucked_weight": shucked_weight,
        "Viscera_weight": viscera_weight, "Shell_weight": shell_weight
    }])
    rings = float(bundle["model"].predict(X_new)[0])
    return {"rings": rings, "age": rings + 1.5}