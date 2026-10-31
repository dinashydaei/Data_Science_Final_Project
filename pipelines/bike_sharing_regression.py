import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "datasets", "bike_sharing_1.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
BUNDLE_PATH = os.path.join(MODEL_DIR, "bike_sharing_regression.pkl")
META_PATH   = os.path.join(MODEL_DIR, "bike_sharing_regression_meta.json")

DATASET_NAME = "Bike Sharing"
TASK         = "regression"
TASK_LABEL   = "Regression"

FEATURES = ["season", "yr", "mnth", "hr", "holiday", "weekday",
            "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"]
TARGET   = "cnt"

NUM_COLS = ["temp", "atemp", "hum", "windspeed", "hr", "mnth", "weekday"]
CAT_COLS = ["season", "yr", "holiday", "workingday", "weathersit"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    drop = [c for c in ["instant", "dteday", "casual", "registered"] if c in df.columns]
    df = df.drop(columns=drop).dropna()
    return df


def _make_model(name):
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=120, max_depth=16, min_samples_leaf=2,
            random_state=42, n_jobs=-1)
    if name == "gb":
        return GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42)
    return Ridge(alpha=1.0)


def _make_pipeline(name):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", "passthrough", CAT_COLS),
    ])
    return Pipeline([("pre", pre), ("model", _make_model(name))])


def _evaluate(y_true, y_pred):
    return {
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 2),
        "R2":   round(float(r2_score(y_true, y_pred)), 4),
    }


def _quality_label(v, th):
    if v < th["low"]:  return "Low"
    if v < th["mid"]:  return "Moderate"
    if v < th["high"]: return "High"
    return "Very High"


def train_and_evaluate(progress_cb=None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    results, models = {}, {}
    for name in ["rf", "gb", "ridge"]:
        if progress_cb:
            progress_cb(f"Training model {name.upper()} ...")
        pipe = _make_pipeline(name)
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        results[name] = _evaluate(y_te, pred)
        models[name] = pipe

    best = max(results, key=lambda k: results[k]["R2"])
    pred_best = models[best].predict(X_te)

    bundle = {
        "model":       models[best],
        "best_model":  best,
        "features":    FEATURES,
        "target":      TARGET,
        "metrics":     results,
        "task":        TASK,
        "test_true":   np.asarray(y_te),
        "test_pred":   np.asarray(pred_best),
    }

    q = np.percentile(y, [25, 60, 85])
    meta = {
        "dataset_name": DATASET_NAME,
        "task": TASK,
        "task_label": TASK_LABEL,
        "target": TARGET,
        "features": FEATURES,
        "metrics": results,
        "best_model": best,
        "quality_thresholds": {"low": float(q[0]), "mid": float(q[1]), "high": float(q[2])},
        "y_min":  float(y.min()),
        "y_max":  float(y.max()),
        "y_mean": float(y.mean()),
        "n_rows": int(len(df)),
    }

    joblib.dump(bundle, BUNDLE_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if progress_cb:
        progress_cb("Training complete ✓")
    return meta


def load_bundle():
    if not (os.path.exists(BUNDLE_PATH) and os.path.exists(META_PATH)):
        return None
    bundle = joblib.load(BUNDLE_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return bundle, meta


def predict_row(bundle, meta, row_dict):
    df = pd.DataFrame([row_dict])[bundle["features"]]
    pred = float(bundle["model"].predict(df)[0])
    pred = max(0.0, pred)
    return pred, _quality_label(pred, meta["quality_thresholds"])