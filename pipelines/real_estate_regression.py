import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

TARGET = "Y house price of unit area"
FEATURES = [
    "X1 transaction date", "X2 house age", "X3 distance to the nearest MRT station",
    "X4 number of convenience stores", "X5 latitude", "X6 longitude"
]


def _app_base():
    if getattr(sys, "frozen", False): return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_PATH = os.path.join(_app_base(), "datasets", "real_estate.csv")
MODEL_PATH = os.path.join(os.getcwd(), "models", "real_estate_model.pkl")


def load_data():
    global TARGET
    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    if TARGET not in df.columns:
        price_cols = [c for c in df.columns if 'price' in c.lower()]
        if price_cols:
            TARGET = price_cols[0]
        else:
            raise KeyError(f"ستون قیمت پیدا نشد: {df.columns.tolist()}")
    return df


def _build_pipeline(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def train_and_evaluate():
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
        "Support Vector Regressor": SVR(kernel="rbf", C=100, gamma=0.1),
        "Ridge Regression": Ridge(alpha=1.0),
        "Linear Regression": LinearRegression()
    }

    results = {}
    best_r2 = -float('inf')
    best_name = ""
    best_pipe = None
    best_y_pred = None

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
            best_y_pred = y_pred

    final_model = _build_pipeline(models[best_name])
    final_model.fit(X, y)

    # ذخیره مدل نهایی + y_test/y_pred برای دکمه Predicted vs Actual
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({
        "model": final_model,
        "metrics": results,
        "best_name": best_name,
        "y_test": y_test.values,
        "y_pred": best_y_pred,
        "feature_names": FEATURES,
    }, MODEL_PATH)

    return results, best_name


def predict(transaction_date, house_age, distance_to_mrt, stores, latitude, longitude):
    if not os.path.exists(MODEL_PATH):
        train_and_evaluate()
    bundle = joblib.load(MODEL_PATH)
    X_new = pd.DataFrame([{
        "X1 transaction date": transaction_date,
        "X2 house age": house_age,
        "X3 distance to the nearest MRT station": distance_to_mrt,
        "X4 number of convenience stores": stores,
        "X5 latitude": latitude,
        "X6 longitude": longitude
    }])
    return {"price": float(bundle["model"].predict(X_new)[0])}