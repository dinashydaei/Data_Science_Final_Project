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

NUMERIC_FEATURES = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL_FEATURES = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]
TARGET = "income"


def _app_base():
    if getattr(sys, "frozen", False):
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_PATH = os.path.join(_app_base(), "datasets", "census_income.csv")
MODEL_DIR = os.path.join(os.getcwd(), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "census_income_model.pkl")


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"دیتاست پیدا نشد:\n{DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]

    df = df.dropna().reset_index(drop=True)

    # تارگت به باینری
    df[TARGET] = df[TARGET].astype(str).str.strip()
    df[TARGET] = df[TARGET].apply(lambda x: 1 if x == ">50K" else 0)

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=NUMERIC_FEATURES + [TARGET]).reset_index(drop=True)

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip()

    return df


def _build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL_FEATURES),
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
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=20,
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


def predict(age, workclass, education, education_num, marital_status,
            occupation, relationship, race, sex, capital_gain,
            capital_loss, hours_per_week, native_country):
    if not os.path.exists(MODEL_PATH):
        train_and_evaluate()
    bundle = joblib.load(MODEL_PATH)

    X_new = pd.DataFrame([{
        "age": int(age),
        "workclass": workclass,
        "education": education,
        "education-num": int(education_num),
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital-gain": float(capital_gain),
        "capital-loss": float(capital_loss),
        "hours-per-week": float(hours_per_week),
        "native-country": native_country,
    }])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    pred = int(bundle["model"].predict(X_new)[0])
    proba = bundle["model"].predict_proba(X_new)[0]
    conf = float(proba[pred])

    label = "Income > 50K 💰" if pred == 1 else "Income ≤ 50K"
    return {"label": label, "value": pred, "confidence": conf}