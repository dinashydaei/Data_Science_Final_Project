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

TARGET = "percent_change_next_weeks_price"
DROP_COLS = ["quarter", "stock", "date", "previous_weeks_volume"]

NUMERIC_FEATURES = [
    "open", "high", "low", "close", "volume",
    "percent_change_price", "percent_change_volume_over_last_wk",
    "next_weeks_open", "next_weeks_close",
    "days_to_next_dividend", "percent_return_next_dividend"
]


def _app_base():
    if getattr(sys, "frozen", False): return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_PATH = os.path.join(_app_base(), "datasets", "dow_jones.csv")
MODEL_DIR = os.path.join(os.getcwd(), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "dow_jones_model.pkl")


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"دیتاست پیدا نشد:\n{DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]

    for col in NUMERIC_FEATURES + [TARGET]:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


def _build_preprocessor():
    # اول مقادیر خالی (NaN) رو پر می‌کنه، بعد مقیاس‌بندی می‌کنه
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])


def train_and_evaluate():
    df = load_data()
    X = df[NUMERIC_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
    }

    results = {}
    best_rmse = float("inf")
    best_name, best_pipe, best_y_pred = "", None, None

    for name, model in models.items():
        # استفاده از preprocessor که NaN رو مدیریت می‌کنه
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


def predict(open_p, high, low, close, volume, pct_price, pct_vol, next_open, next_close, days_div, pct_ret_div):
    if not os.path.exists(MODEL_PATH): train_and_evaluate()
    bundle = joblib.load(MODEL_PATH)

    X_new = pd.DataFrame([{
        "open": float(open_p), "high": float(high), "low": float(low), "close": float(close),
        "volume": float(volume), "percent_change_price": float(pct_price),
        "percent_change_volume_over_last_wk": float(pct_vol),
        "next_weeks_open": float(next_open), "next_weeks_close": float(next_close),
        "days_to_next_dividend": float(days_div), "percent_return_next_dividend": float(pct_ret_div),
    }])

    pred = float(bundle["model"].predict(X_new)[0])
    return {"value": pred}