import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

FEATURES = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
TARGET = "Occupancy"


def _app_base():
    if getattr(sys, "frozen", False):
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_PATH = os.path.join(_app_base(), "datasets", "occupancy.csv")
MODEL_DIR = os.path.join(os.getcwd(), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "room_occupancy_model.pkl")


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"دیتاست پیدا نشد:\n{DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]

    if "date" in df.columns:
        df = df.drop(columns=["date"])

    for col in FEATURES + [TARGET]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)

    return df


def train_and_evaluate():
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=15,
                                                class_weight="balanced", n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }

    results = {}
    best_f1 = -1.0
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

        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))

        results[name] = {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}

        if f1 > best_f1:
            best_f1 = f1
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
        "classes": sorted(y.unique().tolist()),
    }, MODEL_PATH)

    return results, best_name


def predict(temperature, humidity, light, co2, humidity_ratio):
    if not os.path.exists(MODEL_PATH):
        train_and_evaluate()
    bundle = joblib.load(MODEL_PATH)

    X_new = pd.DataFrame([{
        "Temperature": float(temperature),
        "Humidity": float(humidity),
        "Light": float(light),
        "CO2": float(co2),
        "HumidityRatio": float(humidity_ratio),
    }])[FEATURES]

    pred = int(bundle["model"].predict(X_new)[0])
    proba = bundle["model"].predict_proba(X_new)[0]
    conf = float(proba[pred])

    label = "Occupied 🧑‍💼" if pred == 1 else "Empty 🚪"
    return {"label": label, "value": pred, "confidence": conf}