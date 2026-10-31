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

TARGET = "default_payment_next_month"

NUMERIC_FEATURES = ["LIMIT_BAL", "AGE",
                    "BILL_AMT1", "BILL_AMT2",
                    "PAY_AMT1", "PAY_AMT2"]

CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE",
                        "PAY_0", "PAY_2", "PAY_3"]


def _app_base():
    if getattr(sys, "frozen", False):
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_PATH = os.path.join(_app_base(), "datasets", "credit_default.csv")
MODEL_DIR = os.path.join(os.getcwd(), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "credit_default_model.pkl")


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"دیتاست پیدا نشد:\n{DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]

    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]:
        if col not in df.columns:
            raise ValueError(f"ستون «{col}» در CSV پیدا نشد.")

    # عددی‌ها
    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # categorical ها به str (تا OneHotEncoder درست کار کنه)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip()

    df = df.dropna(subset=NUMERIC_FEATURES + [TARGET]).reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)

    return df


def _build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
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

    # نام فیچرها بعد از one-hot
    feature_names = (
        NUMERIC_FEATURES
        + list(final_pipe.named_steps["prep"]
               .named_transformers_["cat"]
               .get_feature_names_out(CATEGORICAL_FEATURES))
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        "model": final_pipe,
        "metrics": results,
        "best_name": best_name,
        "y_test": y_test.values,
        "y_pred": best_y_pred,
        "feature_names": feature_names,
    }, MODEL_PATH)

    return results, best_name


def predict(limit_bal, sex, education, marriage, age,
            pay_0, pay_2, pay_3,
            bill_amt1, bill_amt2,
            pay_amt1, pay_amt2):
    if not os.path.exists(MODEL_PATH):
        train_and_evaluate()
    bundle = joblib.load(MODEL_PATH)

    X_new = pd.DataFrame([{
        "LIMIT_BAL": float(limit_bal),
        "SEX": str(sex),
        "EDUCATION": str(education),
        "MARRIAGE": str(marriage),
        "AGE": float(age),
        "PAY_0": str(pay_0),
        "PAY_2": str(pay_2),
        "PAY_3": str(pay_3),
        "BILL_AMT1": float(bill_amt1),
        "BILL_AMT2": float(bill_amt2),
        "PAY_AMT1": float(pay_amt1),
        "PAY_AMT2": float(pay_amt2),
    }])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    pred = int(bundle["model"].predict(X_new)[0])
    proba = bundle["model"].predict_proba(X_new)[0]
    conf = float(proba[pred])

    label = "Default Risk ⚠️" if pred == 1 else "No Default ✅"
    return {"label": label, "value": pred, "confidence": conf}