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

TARGET = "Target_Total_orders"
FEATURES = [
    "Week_of_month", "Day_of_week", "Non_urgent_order", "Urgent_order",
    "Order_type_A", "Order_type_B", "Order_type_C", "Fiscal_sector_orders",
    "Traffic_controller_orders", "Banking_orders_1", "Banking_orders_2", "Banking_orders_3"
]


def _app_base():
    if getattr(sys, "frozen", False): return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_PATH = os.path.join(_app_base(), "datasets", "daily_demand.csv")
MODEL_PATH = os.path.join(os.getcwd(), "models", "daily_demand_model.pkl")


def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    # تمیزکاری داده‌ها (حذف مقادیر خالی احتمالی)
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
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
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

    # آموزش نهایی با ۱۰۰٪ داده
    final_model = _build_pipeline(models[best_name])
    final_model.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": final_model, "metrics": results, "best_name": best_name}, MODEL_PATH)
    return results, best_name


def predict(week_of_month, day_of_week, non_urgent, urgent, order_a, order_b, order_c, fiscal, traffic, bank1, bank2,
            bank3):
    if not os.path.exists(MODEL_PATH): train_and_evaluate()
    bundle = joblib.load(MODEL_PATH)

    X_new = pd.DataFrame([{
        "Week_of_month": week_of_month, "Day_of_week": day_of_week,
        "Non_urgent_order": non_urgent, "Urgent_order": urgent,
        "Order_type_A": order_a, "Order_type_B": order_b, "Order_type_C": order_c,
        "Fiscal_sector_orders": fiscal, "Traffic_controller_orders": traffic,
        "Banking_orders_1": bank1, "Banking_orders_2": bank2, "Banking_orders_3": bank3
    }])

    total = float(bundle["model"].predict(X_new)[0])
    return {"total": total}