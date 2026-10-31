import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

TARGET = "Class"
FEATURES = ["Variance", "Skewness", "Curtosis", "Entropy"]


def _app_base():
    if getattr(sys, "frozen", False):
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_PATH = os.path.join(_app_base(), "datasets", "bank_note.csv")
MODEL_DIR = os.path.join(os.getcwd(), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "banknote_model.pkl")


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"دیتاست پیدا نشد:\n{DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]

    for col in FEATURES + [TARGET]:
        if col not in df.columns:
            raise ValueError(f"ستون «{col}» در CSV پیدا نشد.")

    for col in FEATURES + [TARGET]:
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
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15,
                                                n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
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
        "feature_names": FEATURES,
    }, MODEL_PATH)

    return results, best_name


def predict(variance, skewness, curtosis, entropy):
    if not os.path.exists(MODEL_PATH):
        train_and_evaluate()
    bundle = joblib.load(MODEL_PATH)

    X_new = pd.DataFrame([{
        "Variance": float(variance),
        "Skewness": float(skewness),
        "Curtosis": float(curtosis),
        "Entropy": float(entropy),
    }])[FEATURES]

    pred = int(bundle["model"].predict(X_new)[0])
    proba = bundle["model"].predict_proba(X_new)[0]
    conf = float(proba[pred])

    label = "Genuine ✅" if pred == 0 else "Forged ⚠️"
    return {"label": label, "value": pred, "confidence": conf}