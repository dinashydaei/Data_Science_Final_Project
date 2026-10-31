import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

TARGET = "Class"
FEATURES = [
    "Industrial_Risk", "Management_Risk", "Financial_Flexibility",
    "Credibility", "Competitiveness", "Operating_Risk"
]


def _app_base():
    if getattr(sys, "frozen", False): return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_PATH = os.path.join(_app_base(), "datasets", "bankruptcy.csv")
MODEL_PATH = os.path.join(os.getcwd(), "models", "bankruptcy_model.pkl")


def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]

    # تمیزکاری داده‌های متنی
    for col in FEATURES + [TARGET]:
        df[col] = df[col].astype(str).str.strip().str.upper()
    return df


def train_and_evaluate():
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42,
                                                        stratify=y_encoded)

    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown='ignore'), FEATURES)]
    )

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=200, max_depth=10, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
        "SVM": SVC(kernel="rbf", random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }

    results = {}
    best_acc = 0
    best_name = ""
    best_pipe = None
    best_y_pred = None

    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        results[name] = {"accuracy": acc, "f1": f1}

        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_pipe = pipe
            best_y_pred = y_pred

    # --- آموزش نهایی با ۱۰۰٪ داده ---
    final_model = Pipeline([("prep", preprocessor), ("model", models[best_name])])
    final_model.fit(X, y_encoded)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({
        "model": final_model, "metrics": results, "best_name": best_name,
        "label_encoder": le, "y_test": y_test, "y_pred": best_y_pred, "classes": le.classes_
    }, MODEL_PATH)

    return results, best_name


def predict(industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk):
    if not os.path.exists(MODEL_PATH): train_and_evaluate()
    bundle = joblib.load(MODEL_PATH)

    X_new = pd.DataFrame([{
        "Industrial_Risk": industrial_risk, "Management_Risk": management_risk,
        "Financial_Flexibility": financial_flexibility, "Credibility": credibility,
        "Competitiveness": competitiveness, "Operating_Risk": operating_risk
    }])

    pred_encoded = bundle["model"].predict(X_new)[0]
    pred_label = bundle["label_encoder"].inverse_transform([pred_encoded])[0]

    final_label = "Non-Bankrupt (NB)" if pred_label == "NB" else "Bankrupt (B)"
    return {"label": final_label}