import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

TARGET = "Confirmed"
FEATURES = ["DayNum", "Lag1", "Lag3", "Lag7", "Rolling7Mean", "DayOfWeek"]


def _app_base():
    if getattr(sys, "frozen", False):
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_PATH = os.path.join(_app_base(), "datasets", "covid_19_data.csv")
MODEL_DIR = os.path.join(os.getcwd(), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "covid19_model.pkl")


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"دیتاست پیدا نشد:\n{DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]

    df["ObservationDate"] = pd.to_datetime(df["ObservationDate"], errors="coerce")
    df["DayNum"] = pd.to_numeric(df["DayNum"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    df = df.dropna().sort_values("DayNum").reset_index(drop=True)

    # فیچرهای lag
    df["Lag1"] = df[TARGET].shift(1)
    df["Lag3"] = df[TARGET].shift(3)
    df["Lag7"] = df[TARGET].shift(7)
    df["Rolling7Mean"] = df[TARGET].shift(1).rolling(7).mean()
    df["DayOfWeek"] = df["ObservationDate"].dt.dayofweek

    df = df.dropna().reset_index(drop=True)
    return df


def train_and_evaluate():
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=15,
                                               n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    }

    results = {}
    best_rmse = float("inf")
    best_name = ""
    best_pipe = None
    best_y_pred = None

    for name, model in models.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])
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

    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", models[best_name]),
    ])
    final_pipe.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        "model": final_pipe,
        "metrics": results,
        "best_name": best_name,
        "y_test": y_test.values,
        "y_pred": best_y_pred,
        "features": FEATURES,
    }, MODEL_PATH)

    return results, best_name


def predict(day_num, lag1, lag3, lag7, rolling7_mean, day_of_week):
    if not os.path.exists(MODEL_PATH):
        train_and_evaluate()
    bundle = joblib.load(MODEL_PATH)

    X_new = pd.DataFrame([{
        "DayNum": float(day_num),
        "Lag1": float(lag1),
        "Lag3": float(lag3),
        "Lag7": float(lag7),
        "Rolling7Mean": float(rolling7_mean),
        "DayOfWeek": int(day_of_week),
    }])[FEATURES]

    pred = float(bundle["model"].predict(X_new)[0])
    pred = max(pred, 0.0)

    if pred < 10_000:
        level = "Low"
    elif pred < 100_000:
        level = "Moderate"
    elif pred < 500_000:
        level = "High"
    else:
        level = "Critical"

    return {"value": pred, "label": level}