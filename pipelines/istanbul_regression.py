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

TARGET = "ISE"
FEATURES = ["SP", "DAX", "FTSE", "NIKKEI", "BOVESPA", "EU", "EM"]


def _app_base():
    if getattr(sys, "frozen", False): return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_PATH = os.path.join(_app_base(), "datasets", "istanbul_stock.csv")
MODEL_DIR = os.path.join(os.getcwd(), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "istanbul_model.pkl")


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"دیتاست پیدا نشد:\n{DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]

    if "date" in df.columns:
        df = df.drop(columns=["date"])

    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

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
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
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


def predict(sp, dax, ftse, nikkei, bovespa, eu, em):
    if not os.path.exists(MODEL_PATH): train_and_evaluate()
    bundle = joblib.load(MODEL_PATH)

    X_new = pd.DataFrame([{
        "SP": float(sp), "DAX": float(dax), "FTSE": float(ftse),
        "NIKKEI": float(nikkei), "BOVESPA": float(bovespa),
        "EU": float(eu), "EM": float(em)
    }])

    pred = float(bundle["model"].predict(X_new)[0])
    return {"value": pred}