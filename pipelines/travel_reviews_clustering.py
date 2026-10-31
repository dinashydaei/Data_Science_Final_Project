import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

FEATURES = [f"Category {i}" for i in range(1, 11)]


def _app_base():
    if getattr(sys, "frozen", False): return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_PATH = os.path.join(_app_base(), "datasets", "travel_reviews.csv")
MODEL_DIR = os.path.join(os.getcwd(), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "travel_reviews_model.pkl")


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


def predict(cat1, cat2, cat3, cat4, cat5, cat6, cat7, cat8, cat9, cat10):
    if not os.path.exists(MODEL_PATH): train_and_evaluate()
    bundle = joblib.load(MODEL_PATH)

    X_new = pd.DataFrame([{
        "Category 1": float(cat1), "Category 2": float(cat2),
        "Category 3": float(cat3), "Category 4": float(cat4),
        "Category 5": float(cat5), "Category 6": float(cat6),
        "Category 7": float(cat7), "Category 8": float(cat8),
        "Category 9": float(cat9), "Category 10": float(cat10)
    }])

    X_scaled = bundle["scaler"].transform(X_new)
    model = bundle["model"]

    # بررسی اینکه مدل متد predict دارد یا نه (مثل K-Means)
    if hasattr(model, "predict"):
        cluster = int(model.predict(X_scaled)[0])
    else:

        from sklearn.cluster import KMeans
        k = len(set(bundle["labels"]))
        fallback_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        fallback_kmeans.fit(bundle["scaler"].transform(pd.read_csv(DATA_PATH)[FEATURES]))
        cluster = int(fallback_kmeans.predict(X_scaled)[0])

    if cluster == 0:
        label = "Cluster 0: High Overall Satisfaction"
    elif cluster == 1:
        label = "Cluster 1: Moderate / Mixed Reviews"
    elif cluster == 2:
        label = "Cluster 2: Low / Specific Category Focus"
    else:
        label = f"Cluster {cluster}"

    return {"label": label, "cluster": cluster}