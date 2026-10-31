import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

TARGET = "Output"
FEATURES = [
    "Season", "Age", "Childish_Diseases", "Trauma", "Surgical_intervention",
    "High_fevers_lastyear", "Alcohol_consumption", "Smoking_Habit", "Hours_sitting"
]


def _app_base():
    if getattr(sys, "frozen", False): return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_PATH = os.path.join(_app_base(), "datasets", "fertility.csv")
MODEL_PATH = os.path.join(os.getcwd(), "models", "fertility_model.pkl")


def load_data():
    global TARGET
    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]

    df[TARGET] = df[TARGET].astype(str).str.strip().str.upper()
    return df


def train_and_evaluate():
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42,
                                                        stratify=y_encoded)

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
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
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

    final_model = Pipeline([("scaler", StandardScaler()), ("model", models[best_name])])
    final_model.fit(X, y_encoded)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({
        "model": final_model,
        "metrics": results,
        "best_name": best_name,
        "label_encoder": le,
        "y_test": y_test,
        "y_pred": best_y_pred,
        "classes": le.classes_
    }, MODEL_PATH)

    return results, best_name


def predict(season, age, childish_diseases, trauma, surgical_intervention,
            high_fevers_lastyear, alcohol_consumption, smoking_habit, hours_sitting):
    if not os.path.exists(MODEL_PATH): train_and_evaluate()
    bundle = joblib.load(MODEL_PATH)

    X_new = pd.DataFrame([{
        "Season": season, "Age": age, "Childish_Diseases": childish_diseases,
        "Trauma": trauma, "Surgical_intervention": surgical_intervention,
        "High_fevers_lastyear": high_fevers_lastyear, "Alcohol_consumption": alcohol_consumption,
        "Smoking_Habit": smoking_habit, "Hours_sitting": hours_sitting
    }])

    pred_encoded = bundle["model"].predict(X_new)[0]
    pred_label = bundle["label_encoder"].inverse_transform([pred_encoded])[0]

    # تبدیل N و O به کلمات خوانا
    final_label = "Normal (N)" if pred_label == "N" else "Altered (O)"
    return {"label": final_label}