# utils/visuals.py
import os
import uuid
import tempfile
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from PIL import Image
import customtkinter as ctk

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
TEXT_DARK = "#1F2937"  # خاکستری تیره
PURPLE = "#7C3AED"  # بنفش اصلی
PURPLE_DARK = "#5B21B6"  # بنفش تیره
PURPLE_LIGHT = "#EDE9FE"  # بنفش خیلی روشن
GREEN = "#059669"  # فقط برای تاکید موفقیت
RED = "#DC2626"  # خطا
WHITE_BG = "#FFFFFF"  # پس‌زمینه کارت‌ها
BG_LIGHT = "#F5F3FF"  # پس‌زمینه بنفش خیلی روشن
GRID_COLOR = "#E9E5FF"  # خطوط شبکه


def _set_light_theme():
    """تم لایت با رنگ‌های سفید/بنفش"""
    plt.style.use('default')
    plt.rcParams.update({
        "figure.facecolor": BG_LIGHT,
        "axes.facecolor": WHITE_BG,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_DARK,
        "axes.titlecolor": PURPLE_DARK,
        "xtick.color": TEXT_DARK,
        "ytick.color": TEXT_DARK,
        "text.color": TEXT_DARK,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.6,
        "figure.dpi": 100,
    })


def _display_plot():
    """ذخیره نمودار فعلی به صورت عکس و نمایش در یک پنجره جدید"""
    plt.tight_layout()
    path = os.path.join(tempfile.gettempdir(), f"plot_{uuid.uuid4().hex}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    win = ctk.CTkToplevel()
    win.title("EDA Plot")
    win.geometry("850x650")
    win.attributes("-topmost", True)

    img = ctk.CTkImage(light_image=Image.open(path), dark_image=Image.open(path), size=(800, 600))
    lbl = ctk.CTkLabel(win, text="", image=img)
    lbl.pack(expand=True, fill="both", padx=10, pady=10)
    lbl._img = img


def plot_heatmap(df):
    _set_light_theme()
    numeric_df = df.copy()

    for col in numeric_df.columns:
        if numeric_df[col].dtype == 'object' or numeric_df[col].dtype.name == 'string':
            le = LabelEncoder()
            numeric_df[col] = le.fit_transform(numeric_df[col].astype(str))

    numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    plt.figure(figsize=(10, 8))
    corr = numeric_df.corr()
    cmap = sns.diverging_palette(265, 150, s=80, l=55, as_cmap=True)
    sns.heatmap(
        corr, annot=True, cmap=cmap, fmt=".2f",
        linewidths=0.5, cbar_kws={'label': 'Correlation'},
        annot_kws={"color": TEXT_DARK, "size": 9}
    )
    plt.title("Correlation Heatmap", fontsize=16, color=PURPLE_DARK, fontweight='bold', pad=15)
    _display_plot()


def plot_confusion_matrix(y_true, y_pred, labels):
    _set_light_theme()
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap="Purples",
        xticklabels=labels, yticklabels=labels,
        annot_kws={"size": 14, "color": TEXT_DARK},
        linewidths=0.8, linecolor=GRID_COLOR
    )
    plt.title("Confusion Matrix", fontsize=16, color=PURPLE_DARK, fontweight='bold', pad=15)
    plt.xlabel("Predicted", color=PURPLE, fontweight='bold')
    plt.ylabel("Actual", color=PURPLE, fontweight='bold')
    _display_plot()


def plot_pairplot(df, target_col=None):
    _set_light_theme()
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    cols = numeric_df.columns[:4]
    plot_df = df[cols].copy()

    if target_col and target_col in df.columns:
        plot_df[target_col] = df[target_col]
        g = sns.pairplot(plot_df, hue=target_col, corner=True, palette="Purples")
    else:
        g = sns.pairplot(plot_df, corner=True, palette="Purples")

    g.fig.suptitle("Pairplot", y=1.02, fontsize=16, color=PURPLE_DARK, fontweight='bold')
    _display_plot()


def plot_elbow_method(df, max_k=10):
    _set_light_theme()
    X = df.select_dtypes(include=['float64', 'int64'])
    wcss = []

    for i in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), wcss, marker='o', color=PURPLE, linewidth=2,
             markerfacecolor=PURPLE_DARK, markersize=8)
    plt.title("Elbow Method", fontsize=16, color=PURPLE_DARK, fontweight='bold', pad=15)
    plt.xlabel("Number of Clusters", fontsize=12)
    plt.ylabel("WCSS", fontsize=12)
    plt.grid(alpha=0.4, color=GRID_COLOR)
    _display_plot()


def plot_categorical_bar_chart(df, target_col=None):
    _set_light_theme()
    cat_cols = df.select_dtypes(include=['object', 'string']).columns

    if len(cat_cols) == 0:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "No categorical data found.",
                 ha='center', va='center', fontsize=14, color=TEXT_DARK)
        plt.axis('off')
        _display_plot()
        return

    n_cols = min(len(cat_cols), 6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    colors = sns.color_palette("Purples", n_colors=5)

    for i, col in enumerate(cat_cols[:n_cols]):
        sns.countplot(data=df, x=col, ax=axes[i], palette=colors, edgecolor=TEXT_DARK)
        axes[i].set_title(col, color=PURPLE_DARK, fontweight='bold', fontsize=14)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Count", color=TEXT_DARK)
        axes[i].tick_params(axis='x', rotation=0)
        axes[i].grid(alpha=0.3, axis='y', color=GRID_COLOR)

    for j in range(i + 1, 6):
        fig.delaxes(axes[j])

    plt.suptitle("Distribution of Categorical Features",
                 fontsize=20, color=PURPLE_DARK, fontweight='bold', y=0.98)
    _display_plot()


def plot_mpg_vs_features(df, target_col="mpg"):
    _set_light_theme()

    if "cylinders" in df.columns and target_col in df.columns:
        plt.figure(figsize=(10, 6))
        grouped = df.groupby("cylinders")[target_col].mean().reset_index()

        bars = plt.bar(
            grouped["cylinders"].astype(str), grouped[target_col],
            color=PURPLE, edgecolor=PURPLE_DARK, linewidth=1.5
        )

        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2, yval + 0.5, round(yval, 1),
                ha='center', va='bottom', color=TEXT_DARK, fontweight='bold'
            )

        plt.title("Average MPG by Cylinder Count",
                  fontsize=16, color=PURPLE_DARK, fontweight='bold', pad=15)
        plt.xlabel("Number of Cylinders", fontsize=12)
        plt.ylabel("Average MPG", fontsize=12)
        plt.grid(axis='y', alpha=0.4, color=GRID_COLOR)
    else:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "Required columns not found.",
                 ha='center', va='center', fontsize=14, color=TEXT_DARK)
        plt.axis('off')

    _display_plot()



#  Pollution (Regression)
def plot_parity(y_true, y_pred, unit="PM2.5 (µg/m³)"):
    _set_light_theme()
    y_true = pd.Series(y_true).astype(float).values
    y_pred = pd.Series(y_pred).astype(float).values

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_true, y_pred, alpha=0.35, s=12, color=PURPLE, edgecolor="none")

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "--", lw=2, color=PURPLE_DARK, label="Ideal (y = x)")

    ax.set_xlabel(f"Actual {unit}", fontsize=12, color=TEXT_DARK)
    ax.set_ylabel(f"Predicted {unit}", fontsize=12, color=TEXT_DARK)
    ax.set_title("Predicted vs Actual", fontsize=16, color=PURPLE_DARK,
                 fontweight="bold", pad=15)
    ax.grid(alpha=0.3, color=GRID_COLOR)
    ax.legend(loc="upper left", framealpha=0.9)

    mae = float(pd.Series(abs(y_true - y_pred)).mean())
    rmse = float((pd.Series((y_true - y_pred) ** 2).mean()) ** 0.5)
    ax.text(
        0.98, 0.02, f"MAE = {mae:.2f}\nRMSE = {rmse:.2f}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=11, color=TEXT_DARK,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=WHITE_BG,
                  edgecolor=GRID_COLOR, alpha=0.95)
    )
    _display_plot()


def plot_residuals(y_true, y_pred):
    _set_light_theme()
    y_true = pd.Series(y_true).astype(float).values
    y_pred = pd.Series(y_pred).astype(float).values
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.histplot(residuals, bins=50, kde=True, ax=axes[0],
                 color=PURPLE, edgecolor=PURPLE_DARK, alpha=0.6)
    axes[0].axvline(0, color=PURPLE_DARK, linestyle="--", linewidth=2, label="Zero")
    axes[0].set_title("Residuals Distribution", fontsize=14,
                      color=PURPLE_DARK, fontweight="bold")
    axes[0].set_xlabel("Residual (Actual − Predicted)", color=TEXT_DARK)
    axes[0].set_ylabel("Frequency", color=TEXT_DARK)
    axes[0].grid(alpha=0.3, color=GRID_COLOR)
    axes[0].legend()

    axes[1].scatter(y_pred, residuals, alpha=0.3, s=10,
                    color=PURPLE, edgecolor="none")
    axes[1].axhline(0, color=PURPLE_DARK, linestyle="--", linewidth=2)
    axes[1].set_title("Residuals vs Predicted", fontsize=14,
                      color=PURPLE_DARK, fontweight="bold")
    axes[1].set_xlabel("Predicted", color=TEXT_DARK)
    axes[1].set_ylabel("Residual", color=TEXT_DARK)
    axes[1].grid(alpha=0.3, color=GRID_COLOR)

    _display_plot()


def plot_pm25_timeseries(df, target_col="pm2.5", n_points=2000):
    _set_light_theme()

    if target_col not in df.columns:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, f"Column '{target_col}' not found.",
                 ha="center", va="center", fontsize=14, color=TEXT_DARK)
        plt.axis("off")
        _display_plot()
        return

    plot_df = df.copy()

    if "datetime" in plot_df.columns:
        plot_df = plot_df.set_index("datetime")
    else:
        plot_df = plot_df.set_index(pd.RangeIndex(len(plot_df)))

    step = max(1, len(plot_df) // n_points)
    plot_df = plot_df.iloc[::step]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(plot_df.index, plot_df[target_col],
            color=PURPLE, linewidth=0.8, alpha=0.85)
    ax.axhline(plot_df[target_col].mean(), color=PURPLE_DARK,
               linestyle="--", linewidth=2,
               label=f"Mean = {plot_df[target_col].mean():.1f}")

    ax.set_title("PM2.5 Over Time", fontsize=16, color=PURPLE_DARK,
                 fontweight="bold", pad=15)
    ax.set_xlabel("Time", fontsize=12, color=TEXT_DARK)
    ax.set_ylabel("PM2.5 (µg/m³)", fontsize=12, color=TEXT_DARK)
    ax.grid(alpha=0.3, color=GRID_COLOR)
    ax.legend()
    _display_plot()


def plot_pm25_monthly_box(df, target_col="pm2.5", month_col="month"):
    _set_light_theme()

    if target_col not in df.columns or month_col not in df.columns:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "Required columns not found.",
                 ha="center", va="center", fontsize=14, color=TEXT_DARK)
        plt.axis("off")
        _display_plot()
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=df, x=month_col, y=target_col, ax=ax,
        palette="Purples", showfliers=False, linewidth=1.2
    )
    ax.set_title("PM2.5 Distribution by Month", fontsize=16,
                 color=PURPLE_DARK, fontweight="bold", pad=15)
    ax.set_xlabel("Month", fontsize=12, color=TEXT_DARK)
    ax.set_ylabel("PM2.5 (µg/m³)", fontsize=12, color=TEXT_DARK)
    ax.grid(alpha=0.3, color=GRID_COLOR, axis="y")
    _display_plot()


def plot_pm25_vs_weather(df, target_col="pm2.5",
                         weather_cols=("TEMP", "PRES", "Iws", "DEWP")):
    _set_light_theme()

    available = [c for c in weather_cols if c in df.columns]
    if target_col not in df.columns or not available:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "Required columns not found.",
                 ha="center", va="center", fontsize=14, color=TEXT_DARK)
        plt.axis("off")
        _display_plot()
        return

    n = len(available)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, col in enumerate(available):
        axes[i].scatter(df[col], df[target_col],
                        alpha=0.2, s=6, color=PURPLE, edgecolor="none")
        axes[i].set_title(f"PM2.5 vs {col}", fontsize=13,
                          color=PURPLE_DARK, fontweight="bold")
        axes[i].set_xlabel(col, fontsize=11, color=TEXT_DARK)
        axes[i].set_ylabel("PM2.5 (µg/m³)", fontsize=11, color=TEXT_DARK)
        axes[i].grid(alpha=0.3, color=GRID_COLOR)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("PM2.5 vs Weather Features", fontsize=18,
                 color=PURPLE_DARK, fontweight="bold", y=1.00)
    _display_plot()


def plot_feature_importance(model, feature_names, top_n=15):
    _set_light_theme()
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = np.array(model.feature_importances_)
    elif hasattr(model, "coef_"):
        importances = np.abs(np.array(model.coef_).ravel())
    else:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "Model has no feature importances.",
                 ha="center", va="center", fontsize=14, color=TEXT_DARK)
        plt.axis("off")
        _display_plot()
        return

    idx = importances.argsort()[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    values = importances[idx]

    fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.35)))
    bars = ax.barh(range(len(names)), values,
                   color=PURPLE, edgecolor=PURPLE_DARK, linewidth=1.0)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, color=TEXT_DARK)
    ax.invert_yaxis()
    ax.set_xlabel("Importance", fontsize=12, color=TEXT_DARK)
    ax.set_title("Feature Importance", fontsize=16,
                 color=PURPLE_DARK, fontweight="bold", pad=15)
    ax.grid(alpha=0.3, color=GRID_COLOR, axis="x")

    for bar in bars:
        w = bar.get_width()
        ax.text(w, bar.get_y() + bar.get_height() / 2, f" {w:.3f}",
                va="center", color=TEXT_DARK, fontsize=9)

    _display_plot()
def plot_class_feature_distributions(df, target_col, feature_cols, class_labels=None):
    """
    KDE plot توزیع هر فیچر به تفکیک کلاس.
    مناسب برای دیدن اینکه هر فیچر چقدر کلاس‌ها رو از هم جدا می‌کنه.
    """
    _set_light_theme()

    available = [c for c in feature_cols if c in df.columns]
    if target_col not in df.columns or not available:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "Required columns not found.",
                 ha="center", va="center", fontsize=14, color=TEXT_DARK)
        plt.axis("off")
        _display_plot()
        return

    n = len(available)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    classes = sorted(df[target_col].unique().tolist())
    if class_labels is None:
        class_labels = {c: f"Class {c}" for c in classes}

    palette = [PURPLE, "#EF4444", "#10B981", "#F59E0B"]

    for i, col in enumerate(available):
        for j, cls in enumerate(classes):
            subset = df[df[target_col] == cls][col]
            sns.kdeplot(
                subset, ax=axes[i],
                color=palette[j % len(palette)],
                fill=True, alpha=0.35,
                linewidth=1.5,
                label=class_labels.get(cls, f"Class {cls}"),
            )
        axes[i].set_title(f"{col} by {target_col}", fontsize=13,
                          color=PURPLE_DARK, fontweight="bold")
        axes[i].set_xlabel(col, fontsize=11, color=TEXT_DARK)
        axes[i].set_ylabel("Density", fontsize=11, color=TEXT_DARK)
        axes[i].grid(alpha=0.3, color=GRID_COLOR)
        axes[i].legend(loc="best", framealpha=0.9)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Feature Distributions by {target_col}",
                 fontsize=16, color=PURPLE_DARK, fontweight="bold", y=1.00)
    _display_plot()
def plot_scatter_vs_target(df, feature_col, target_col, xlabel=None, ylabel=None):
    """
    Scatter plot یک فیچر در برابر تارگت، برای دیدن رابطه مستقیم.
    مناسب برای regression.
    """
    _set_light_theme()

    if feature_col not in df.columns or target_col not in df.columns:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "Required columns not found.",
                 ha="center", va="center", fontsize=14, color=TEXT_DARK)
        plt.axis("off")
        _display_plot()
        return

    plot_df = df[[feature_col, target_col]].dropna()

    plt.figure(figsize=(10, 6))
    plt.scatter(plot_df[feature_col], plot_df[target_col],
                alpha=0.5, s=25, color=PURPLE, edgecolor=PURPLE_DARK, linewidth=0.4)

    plt.title(f"{target_col} vs {feature_col}",
              fontsize=16, color=PURPLE_DARK, fontweight="bold", pad=15)
    plt.xlabel(xlabel or feature_col, fontsize=12, color=TEXT_DARK)
    plt.ylabel(ylabel or target_col, fontsize=12, color=TEXT_DARK)
    plt.grid(alpha=0.3, color=GRID_COLOR)
    _display_plot()