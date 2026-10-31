import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

TARGET = "Y1"
FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]


def _app_base():
    if getattr(sys, "frozen", False): return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_PATH = os.path.join(_app_base(), "datasets", "energy.csv")
MODEL_DIR = os.path.join(os.getcwd(), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "energy_model.pkl")


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"دیتاست پیدا نشد:\n{DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]

    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    return df


def _build_preprocessor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])


def train_and_evaluate():
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=4,
                                                       random_state=42),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
    }

    results = {}
    best_rmse = float("inf")
    best_name, best_pipe, best_y_pred = "", None, None

    for name, model in models.items():
        pipe = Pipeline([("prep", _build_preprocessor()), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        results[name] = {"rmse": rmse, "mae": mae, "r2": r2}

        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_pipe = pipe
            best_y_pred = y_pred

    final_model = Pipeline([("prep", _build_preprocessor()), ("model", models[best_name])])
    final_model.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        "model": final_model, "metrics": results, "best_name": best_name,
        "y_test": y_test, "y_pred": best_y_pred,
    }, MODEL_PATH)

    return results, best_name


def predict(relative_compactness, surface_area, wall_area, roof_area,
            overall_height, orientation, glazing_area, glazing_area_dist):
    if not os.path.exists(MODEL_PATH): train_and_evaluate()
    bundle = joblib.load(MODEL_PATH)

    X_new = pd.DataFrame([{
        "X1": float(relative_compactness),
        "X2": float(surface_area),
        "X3": float(wall_area),
        "X4": float(roof_area),
        "X5": float(overall_height),
        "X6": int(orientation),
        "X7": float(glazing_area),
        "X8": int(glazing_area_dist),
    }])

    pred = float(bundle["model"].predict(X_new)[0])
    pred = max(pred, 0.0)

    return {"value": pred}