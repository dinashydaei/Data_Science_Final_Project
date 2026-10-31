import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, \
    HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

TARGET = "Concrete_compressive_strength"
FEATURES = [
    "Cement", "Blast_Furnace_Slag", "Fly_Ash", "Water",
    "Superplasticizer", "Coarse_Aggregate", "Fine_Aggregate", "Age"
]


def _app_base():
    if getattr(sys, "frozen", False): return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_PATH = os.path.join(_app_base(), "datasets", "concrete.csv")
MODEL_DIR = os.path.join(os.getcwd(), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "concrete_model.pkl")


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"دیتاست پیدا نشد:\n{DATA_PATH}\n\nفایل concrete.csv رو در پوشه datasets/ قرار بده.")

    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]

    # پاکسازی مقادیر خالی احتمالی
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    return df


def train_and_evaluate():
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=200, max_depth=15, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=300, random_state=42),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
    }

    results = {}
    best_rmse = float("inf")
    best_name = ""
    best_pipe = None
    best_y_pred = None

    for name, model in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
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

    final_model = Pipeline([("scaler", StandardScaler()), ("model", models[best_name])])
    final_model.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        "model": final_model, "metrics": results, "best_name": best_name,
        "y_test": y_test, "y_pred": best_y_pred,
        "trained_at": pd.Timestamp.now().isoformat(),
    }, MODEL_PATH)

    return results, best_name


def predict(cement, blast_furnace_slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age):
    if not os.path.exists(MODEL_PATH): train_and_evaluate()
    bundle = joblib.load(MODEL_PATH)

    X_new = pd.DataFrame([{
        "Cement": float(cement),
        "Blast_Furnace_Slag": float(blast_furnace_slag),
        "Fly_Ash": float(fly_ash),
        "Water": float(water),
        "Superplasticizer": float(superplasticizer),
        "Coarse_Aggregate": float(coarse_aggregate),
        "Fine_Aggregate": float(fine_aggregate),
        "Age": float(age)
    }])

    pred = float(bundle["model"].predict(X_new)[0])
    pred = max(pred, 0.0)  # استحکام منفی معنی نداره

    return {"value": pred}