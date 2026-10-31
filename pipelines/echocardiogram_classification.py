import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

TARGET = "Alive-at-1"
DROP_COLS = ["Survival", "Still-alive", "Name"]

# همه ویژگی‌های اصلی برگردانده شدند تا فرم و مدل همخوانی داشته باشند
NUMERIC_FEATURES = [
    "Age-at-heart-attack",
    "Fractional-shortening",
    "Epss",
    "Lvdd",
    "Wall-motion-score",
    "Wall-motion-index",
    "Mult",
]
CATEGORICAL_FEATURES = ["Pericardial-effusion", "Group"]


def _app_base():
    if getattr(sys, "frozen", False): return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_PATH = os.path.join(_app_base(), "datasets", "echocardiogram.csv")
MODEL_DIR = os.path.join(os.getcwd(), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "echocardiogram_model.pkl")


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"دیتاست پیدا نشد:\n{DATA_PATH}")

    # اضافه کردن on_bad_lines برای جلوگیری از خطای کاماهای اضافی
    df = pd.read_csv(DATA_PATH, na_values=["?"], on_bad_lines="skip")
    df.columns = [str(c).strip() for c in df.columns]

    for col in DROP_COLS:
        if col in df.columns: df = df.drop(columns=[col])

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).astype(str)

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

    # تنظیمات دقیق‌تر برای دیتای کوچک
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_split=4, random_state=42, class_weight="balanced"
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, max_depth=5, min_samples_split=4, random_state=42, class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, C=0.5, random_state=42, class_weight="balanced"
        ),
    }

    results = {}
    best_acc = -1.0
    best_name = ""
    best_pipe = None
    best_y_pred = None
    best_y_proba = None

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

    # آموزش نهایی روی کل داده
    final_model = Pipeline([("prep", _build_preprocessor()), ("model", models[best_name])])
    final_model.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "metrics": results,
            "best_name": best_name,
            "y_test": y_test,
            "y_pred": best_y_pred,
            "y_proba": best_y_proba,
            "classes": np.array(["No", "Yes"]),
            "trained_at": pd.Timestamp.now().isoformat(),
        },
        MODEL_PATH,
    )

    return results, best_name


def predict(age, pericardial_effusion, fractional_shortening, epss, lvdd, wall_motion_score, wall_motion_index, mult,
            group):
    if not os.path.exists(MODEL_PATH): train_and_evaluate()
    bundle = joblib.load(MODEL_PATH)

    # حالا همه فیلدها دقیقاً با مدل همخوانی دارند
    X_new = pd.DataFrame([{
        "Age-at-heart-attack": float(age),
        "Pericardial-effusion": str(int(pericardial_effusion)),
        "Fractional-shortening": float(fractional_shortening),
        "Epss": float(epss),
        "Lvdd": float(lvdd),
        "Wall-motion-score": float(wall_motion_score),
        "Wall-motion-index": float(wall_motion_index),
        "Mult": float(mult),
        "Group": str(int(group)),
    }])

    pred = int(bundle["model"].predict(X_new)[0])
    proba = bundle["model"].predict_proba(X_new)[0]
    confidence = float(proba[pred]) * 100

    label = "Will Survive 1 Year" if pred == 1 else "Won't Survive 1 Year"
    return {"label": label, "confidence": confidence, "pred": pred}