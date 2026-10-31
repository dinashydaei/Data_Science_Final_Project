import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

TARGET = "Class"
FEATURES = [
    "AGE", "SEX", "STEROID", "ANTIVIRALS", "FATIGUE", "MALAISE", "ANOREXIA",
    "LIVER_BIG", "LIVER_FIRM", "SPLEEN_PALPABLE", "SPIDERS", "ASCITES", "VARICES",
    "BILIRUBIN", "ALK_PHOSPHATE", "SGOT", "ALBUMIN", "PROTIME", "HISTOLOGY"
]


def _app_base():
    if getattr(sys, "frozen", False): return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_PATH = os.path.join(_app_base(), "datasets", "hepatitis.csv")
MODEL_DIR = os.path.join(os.getcwd(), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "hepatitis_model.pkl")


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"دیتاست پیدا نشد:\n{DATA_PATH}")

    df = pd.read_csv(DATA_PATH, na_values=["?"])
    df.columns = [str(c).strip() for c in df.columns]

    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    for col in FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TARGET] = df[TARGET].astype(int)
    return df


def _build_preprocessor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])


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
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced"),
    }

    results = {}
    best_acc = -1.0
    best_name, best_pipe, best_y_pred, best_y_proba = "", None, None, None

    for name, model in models.items():
        try:
            pipe = Pipeline([("prep", _build_preprocessor()), ("model", model)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            try:
                auc = roc_auc_score(y_test, y_proba)
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

    final_model = Pipeline([("prep", _build_preprocessor()), ("model", models[best_name])])
    final_model.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        "model": final_model, "metrics": results, "best_name": best_name,
        "y_test": y_test, "y_pred": best_y_pred, "y_proba": best_y_proba,
        "classes": np.array(["Die", "Live"]),  # 1=Live, 2=Die در دیتاست اصلی
    }, MODEL_PATH)

    return results, best_name


def predict(age, sex, steroid, antivirals, fatigue, malaise, anorexia, liver_big, liver_firm,
            spleen_palpable, spiders, ascites, varices, bilirubin, alk_phosphate, sgot, albumin, protime, histology):
    if not os.path.exists(MODEL_PATH): train_and_evaluate()
    bundle = joblib.load(MODEL_PATH)

    X_new = pd.DataFrame([{
        "AGE": float(age), "SEX": int(sex), "STEROID": int(steroid), "ANTIVIRALS": int(antivirals),
        "FATIGUE": int(fatigue), "MALAISE": int(malaise), "ANOREXIA": int(anorexia),
        "LIVER_BIG": int(liver_big), "LIVER_FIRM": int(liver_firm), "SPLEEN_PALPABLE": int(spleen_palpable),
        "SPIDERS": int(spiders), "ASCITES": int(ascites), "VARICES": int(varices),
        "BILIRUBIN": float(bilirubin), "ALK_PHOSPHATE": float(alk_phosphate), "SGOT": float(sgot),
        "ALBUMIN": float(albumin), "PROTIME": float(protime), "HISTOLOGY": int(histology)
    }])

    pred = int(bundle["model"].predict(X_new)[0])
    proba = bundle["model"].predict_proba(X_new)[0]

    # در این دیتاست 1 = Live (زنده ماند) و 2 = Die (فوت)
    confidence = float(proba[0] if pred == 1 else proba[1]) * 100
    label = "Will Live (Survive)" if pred == 1 else "Will Die (Not Survive)"

    return {"label": label, "confidence": confidence, "pred": pred}