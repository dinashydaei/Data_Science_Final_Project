import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

FEATURES = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]


def _app_base():
    if getattr(sys, "frozen", False): return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_PATH = os.path.join(_app_base(), "datasets", "wholesale.csv")
MODEL_DIR = os.path.join(os.getcwd(), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "wholesale_model.pkl")


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"دیتاست پیدا نشد:\n{DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]

    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    return df


def train_and_evaluate():
    df = load_data()
    X = df[FEATURES]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "K-Means (k=3)": KMeans(n_clusters=3, random_state=42, n_init=10),
        "K-Means (k=4)": KMeans(n_clusters=4, random_state=42, n_init=10),
        "K-Means (k=5)": KMeans(n_clusters=5, random_state=42, n_init=10),
        "Agglomerative": AgglomerativeClustering(n_clusters=3),
    }

    results = {}
    best_score = -1.0
    best_name, best_model = "", None
    best_labels = None

    for name, model in models.items():
        try:
            labels = model.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            results[name] = {"silhouette": score, "n_clusters": len(set(labels))}

            if score > best_score:
                best_score = score
                best_name = name
                best_model = model
                best_labels = labels
        except Exception as e:
            results[name] = {"silhouette": None, "error": str(e)}

    if best_model is None:
        raise RuntimeError("هیچ مدلی نتونست خوشه‌بندی کنه.")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        "model": best_model, "scaler": scaler, "metrics": results,
        "best_name": best_name, "labels": best_labels,
    }, MODEL_PATH)

    return results, best_name


def predict(channel, region, fresh, milk, grocery, frozen, detergents, delicassen):
    if not os.path.exists(MODEL_PATH): train_and_evaluate()
    bundle = joblib.load(MODEL_PATH)

    X_new = pd.DataFrame([{
        "Fresh": float(fresh), "Milk": float(milk), "Grocery": float(grocery),
        "Frozen": float(frozen), "Detergents_Paper": float(detergents), "Delicassen": float(delicassen)
    }])

    X_scaled = bundle["scaler"].transform(X_new)
    cluster = int(bundle["model"].predict(X_scaled)[0])

    if cluster == 0:
        label = "Cluster 0: High Retail/Food (Fresh & Milk)"
    elif cluster == 1:
        label = "Cluster 1: High Grocery & Detergents"
    elif cluster == 2:
        label = "Cluster 2: Small / Mixed Orders"
    else:
        label = f"Cluster {cluster}"

    return {"label": label, "cluster": cluster}