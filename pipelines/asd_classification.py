import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

AQ_FEATURES = [f"A{i}_Score" for i in range(1, 11)]
NUMERIC_FEATURES = AQ_FEATURES + ["age"]
CATEGORICAL_FEATURES = [
    "gender", "ethnicity", "jundice", "austim",   # ← jundice (بدون a) - مطابق CSV
    "contry_of_res", "used_app_before", "relation",
]
TARGET = "Class/ASD"


def _app_base():
    if getattr(sys, "frozen", False):
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_PATH = os.path.join(_app_base(), "datasets", "autism.csv")
MODEL_DIR = os.path.join(os.getcwd(), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "asd_model.pkl")


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"دیتاست پیدا نشد:\n{DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # ── پاکسازی نام ستون‌ها ──
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("\r", "", regex=False)
        .str.replace("\n", "", regex=False)
        .str.strip()
    )

    # ── چک وجود ستون‌های ضروری ──
    required = set(AQ_FEATURES + ["age"] + CATEGORICAL_FEATURES + [TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"ستون‌های زیر در CSV پیدا نشدن:\n{missing}\n\n"
            f"ستون‌های موجود:\n{list(df.columns)}"
        )

    # drop نشتی اطلاعات و ستون بی‌فایده
    for drop_col in ["result", "age_desc"]:
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])

    # ? → Unknown
    for col in df.select_dtypes(include="object").columns:
        df[col] = (
            df[col].astype(str)
            .str.strip()
            .replace({"?": "Unknown", "'?": "Unknown", "nan": "Unknown"})
        )

    # age
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age"] = df["age"].fillna(df["age"].median())
    df["age"] = df["age"].clip(1, 100)

    # A1..A10 به عدد
    for col in AQ_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # target
    df[TARGET] = df[TARGET].astype(str).str.upper().str.strip()
    df[TARGET] = df[TARGET].apply(lambda x: 1 if x == "YES" else 0)

    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    return df


def _build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def train_and_evaluate():
    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15,
                                                class_weight="balanced", n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
    }

    results = {}
    best_f1 = -1.0
    best_name = ""
    best_pipe = None
    best_y_pred = None

    for name, model in models.items():
        pipe = Pipeline([
            ("prep", _build_preprocessor()),
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
        ("prep", _build_preprocessor()),
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
    }, MODEL_PATH)

    return results, best_name


def predict(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10,
            age, gender, ethnicity, jaundice, austim,
            contry_of_res, used_app_before, relation):
    if not os.path.exists(MODEL_PATH):
        train_and_evaluate()
    bundle = joblib.load(MODEL_PATH)


    row = {
        "A1_Score": int(a1), "A2_Score": int(a2), "A3_Score": int(a3), "A4_Score": int(a4),
        "A5_Score": int(a5), "A6_Score": int(a6), "A7_Score": int(a7), "A8_Score": int(a8),
        "A9_Score": int(a9), "A10_Score": int(a10),
        "age": float(age),
        "gender": gender, "ethnicity": ethnicity,
        "jundice": jaundice, "austim": austim,
        "contry_of_res": contry_of_res,
        "used_app_before": used_app_before,
        "relation": relation,
    }
    X_new = pd.DataFrame([row])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    pred = int(bundle["model"].predict(X_new)[0])
    proba = bundle["model"].predict_proba(X_new)[0]
    conf = float(proba[pred])

    label = "ASD Positive 🧠" if pred == 1 else "ASD Negative"
    return {"label": label, "value": pred, "confidence": conf}