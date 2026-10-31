import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

TARGET = "Type_of_Glass"
FEATURES = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]


def _app_base():
    if getattr(sys, "frozen", False): return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_PATH = os.path.join(_app_base(), "datasets", "glass.csv")
MODEL_DIR = os.path.join(os.getcwd(), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "glass_model.pkl")


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"دیتاست پیدا نشد:\n{DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]

    if "Id_number" in df.columns:
        df = df.drop(columns=["Id_number"])

    # پاکسازی
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

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
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42,
                                                class_weight="balanced"),
        "Extra Trees": ExtraTreesClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced"),
    }

    results = {}
    best_acc = -1.0
    best_name, best_pipe, best_y_pred, best_y_proba = "", None, None, None

    for name, model in models.items():
        try:
            pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            try:
                auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
            except:
                auc = None

            results[name] = {"accuracy": acc, "f1": f1, "auc": auc}

            if acc > best_acc:
                best_acc = acc
                best_name = name
                best_pipe = pipe
                best_y_pred = y_pred
                best_y_proba = y_proba
        except Exception as e:
            results[name] = {"accuracy": None, "f1": None, "auc": None, "error": str(e)}

    if best_pipe is None:
        raise RuntimeError("هیچ مدلی نتونست آموزش ببینه.")

    final_model = Pipeline([("scaler", StandardScaler()), ("model", models[best_name])])
    final_model.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        "model": final_model, "metrics": results, "best_name": best_name,
        "y_test": y_test, "y_pred": best_y_pred, "y_proba": best_y_proba,
        "classes": np.unique(y),
    }, MODEL_PATH)

    return results, best_name


def predict(ri, na, mg, al, si, k, ca, ba, fe):
    if not os.path.exists(MODEL_PATH): train_and_evaluate()
    bundle = joblib.load(MODEL_PATH)

    X_new = pd.DataFrame([{
        "RI": float(ri), "Na": float(na), "Mg": float(mg), "Al": float(al),
        "Si": float(si), "K": float(k), "Ca": float(ca), "Ba": float(ba), "Fe": float(fe)
    }])

    pred = int(bundle["model"].predict(X_new)[0])

    classes = bundle["classes"]
    if pred in classes:
        idx = list(classes).index(pred)
        confidence = float(bundle["model"].predict_proba(X_new)[0][idx]) * 100
    else:
        confidence = 0.0

    label = f"Type {pred}"
    return {"label": label, "confidence": confidence, "pred": pred}