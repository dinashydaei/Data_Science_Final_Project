import os
import sys
import traceback
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# تنظیمات دیتاست
TARGET = "pm2.5"
NUMERIC_FEATURES = ["DEWP", "TEMP", "PRES", "Iws", "Is", "Ir", "month", "hour", "dayofweek"]
CATEGORICAL_FEATURES = ["cbwd"]

# فقط مدل‌های درختی واجد انتخاب بهترین هستن
# (تا feature_importances_ معنادار داشته باشیم)
TREE_BASED = {
    "Random Forest",
    "Extra Trees",
    "Gradient Boosting",
    "Hist Gradient Boosting",
}


def _app_base():
    """مسیر پایه برای فایل‌های همراه (وقتی exe ساخته می‌شه)"""
    if getattr(sys, "frozen", False):
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_PATH = os.path.join(_app_base(), "datasets", "pollution.csv")
MODEL_DIR = os.path.join(os.getcwd(), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "pollution_model.pkl")



# بارگذاری داده + پاکسازی
def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"دیتاست پیدا نشد:\n{DATA_PATH}\n\n"
            f"فایل pollution.csv رو در پوشه datasets/ قرار بده."
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]

    # ساخت datetime
    if all(c in df.columns for c in ["year", "month", "day", "hour"]):
        df["datetime"] = pd.to_datetime(
            df[["year", "month", "day", "hour"]], errors="coerce"
        )
        df = df.sort_values("datetime").reset_index(drop=True)
    else:
        df["datetime"] = pd.RangeIndex(len(df))

    # ستون‌های موردنیاز
    if "dayofweek" not in df.columns:
        df["dayofweek"] = df["datetime"].dt.dayofweek

    if "month" not in df.columns:
        df["month"] = df["datetime"].dt.month
    if "hour" not in df.columns:
        df["hour"] = df["datetime"].dt.hour

    # پاکسازی target
    df["pm2.5"] = pd.to_numeric(df["pm2.5"], errors="coerce")
    df = df.dropna(subset=["pm2.5"]).reset_index(drop=True)

    # cbwd -> str تمیز
    if "cbwd" in df.columns:
        df["cbwd"] = df["cbwd"].astype(str).str.strip()

    # پر کردن NaN در ویژگی‌های عددی با میانه
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # اگر cbwd NaN داشت
    if "cbwd" in df.columns:
        df["cbwd"] = df["cbwd"].fillna("CV")

    return df



# preprocessor
def _build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )



# آموزش و ارزیابی
def train_and_evaluate():
    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].astype(float)

    if len(df) < 50:
        raise ValueError("داده کافی برای آموزش وجود ندارد (کمتر از ۵۰ ردیف).")

    # split زمانی (نه تصادفی)
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=15, n_jobs=-1, random_state=42
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=200, max_depth=15, n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, random_state=42
        ),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=300, random_state=42
        ),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
    }

    results = {}
    best_rmse = float("inf")
    best_name = ""
    best_pipe = None
    best_y_pred = None

    for name, model in models.items():
        try:
            pipe = Pipeline([("prep", _build_preprocessor()), ("model", model)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))

            results[name] = {"rmse": rmse, "mae": mae, "r2": r2}

            if rmse < best_rmse and name in TREE_BASED:
                best_rmse = rmse
                best_name = name
                best_pipe = pipe
                best_y_pred = y_pred
        except Exception as e:
            results[name] = {"rmse": None, "mae": None, "r2": None, "error": str(e)}

    if best_pipe is None:
        raise RuntimeError(
            "هیچ مدلی نتونست با موفقیت آموزش ببینه. "
            "لاگ خطاها رو بررسی کن."
        )

    # آموزش نهایی روی کل داده
    final_model = Pipeline(
        [("prep", _build_preprocessor()), ("model", models[best_name])]
    )
    final_model.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "metrics": results,
            "best_name": best_name,
            "y_test": y_test,
            "y_pred": best_y_pred,
            "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
            "trained_at": pd.Timestamp.now().isoformat(),
            "n_train": len(X_train),
            "n_test": len(X_test),
        },
        MODEL_PATH,
    )

    return results, best_name


# پیش‌بینی
def _aqi_level(value: float) -> str:
    if value <= 35:
        return "Good"
    elif value <= 75:
        return "Moderate"
    elif value <= 115:
        return "Unhealthy (Sensitive)"
    elif value <= 150:
        return "Unhealthy"
    else:
        return "Hazardous"


def predict(dewp, temp, pres, iws, is_, ir, month, hour, dayofweek, cbwd):
    if not os.path.exists(MODEL_PATH):
        train_and_evaluate()

    bundle = joblib.load(MODEL_PATH)

    X_new = pd.DataFrame(
        [
            {
                "DEWP": float(dewp),
                "TEMP": float(temp),
                "PRES": float(pres),
                "Iws": float(iws),
                "Is": float(is_),
                "Ir": float(ir),
                "month": int(month),
                "hour": int(hour),
                "dayofweek": int(dayofweek),
                "cbwd": str(cbwd),
            }
        ]
    )

    pred = float(bundle["model"].predict(X_new)[0])
    pred = max(pred, 0.0)

    return {"value": pred, "label": _aqi_level(pred)}