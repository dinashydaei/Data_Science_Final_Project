import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

TARGET = "mpg"
FEATURES = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"]


def _app_base():
    if getattr(sys, "frozen", False): return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_PATH = os.path.join(_app_base(), "datasets", "auto_mpg.csv")
MODEL_PATH = os.path.join(os.getcwd(), "models", "auto_mpg_model.pkl")


def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]

    # تبدیل مقادیر متنی به عددی (تبدیل ? به NaN و سپس پر کردن با میانه)
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors='coerce')
    df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())

    # حذف ردیف‌هایی که هنوز NaN دارند (برای اطمینان)
    df = df.dropna(subset=FEATURES + [TARGET])

    return df


def _build_pipeline(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def train_and_evaluate():
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
        "Support Vector Regressor": SVR(kernel="rbf", C=100, gamma=0.1),
        "Ridge Regression": Ridge(alpha=1.0),
        "Linear Regression": LinearRegression()
    }

    results = {}
    best_r2 = -float('inf')
    best_name = ""
    best_pipe = None

    for name, model in models.items():
        pipe = _build_pipeline(model)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        r2 = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))

        results[name] = {"r2": r2, "rmse": rmse, "mae": mae}

        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_pipe = pipe

    final_model = _build_pipeline(models[best_name])
    final_model.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": final_model, "metrics": results, "best_name": best_name}, MODEL_PATH)
    return results, best_name


def predict(cylinders, displacement, horsepower, weight, acceleration, model_year, origin):
    if not os.path.exists(MODEL_PATH): train_and_evaluate()
    bundle = joblib.load(MODEL_PATH)

    # اطمینان از اینکه هیچ مقدار NaN در ورودی کاربر وجود ندارد
    X_new = pd.DataFrame([{
        "cylinders": cylinders, "displacement": displacement, "horsepower": horsepower,
        "weight": weight, "acceleration": acceleration, "model_year": model_year, "origin": origin
    }])

    # اگر ورودی خالی بود، خطا بده
    if X_new.isnull().values.any():
        raise ValueError("Input contains missing values. Please fill all fields.")

    mpg = float(bundle["model"].predict(X_new)[0])
    return {"mpg": mpg}