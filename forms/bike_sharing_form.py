import os
import threading
import random
import tkinter as tk
from tkinter import messagebox, ttk
import customtkinter as ctk
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import tempfile

from pipelines import bike_sharing_regression as pipeline

BG_MAIN = ("#F5F3FF", "#0A0A10")
CARD_BG = ("#FFFFFF", "#151520")
CARD_SHADOW = ("#E9E5FF", "#000000")
BORDER_COLOR = ("#EDE9FE", "#2A2A35")
PURPLE = ("#7C3AED", "#8B5CF6")
PURPLE_DARK = ("#5B21B6", "#6D28D9")
PURPLE_HOVER = ("#6D28D9", "#7C3AED")
TEXT_MAIN = ("#1F2937", "#FFFFFF")
TEXT_MUTED = ("#6B7280", "#9CA3AF")
GREEN = ("#059669", "#10B981")
RED = ("#DC2626", "#EF4444")
AMBER = ("#D97706", "#F59E0B")
ORANGE = ("#EA580C", "#FB923C")

LABEL_MAP = {"rf": "Random Forest", "gb": "Gradient Boosting", "ridge": "Ridge"}


class ShadowCard(ctk.CTkFrame):
    def __init__(self, parent, corner_radius=20, shadow_size=4, fg_color=CARD_BG, border_width=1,
                 border_color=BORDER_COLOR, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.shadow = ctk.CTkFrame(self, fg_color=CARD_SHADOW, corner_radius=corner_radius)
        self.shadow.place(relx=0, rely=0, relwidth=1, relheight=1, x=shadow_size, y=shadow_size)
        self.card = ctk.CTkFrame(self, fg_color=fg_color, corner_radius=corner_radius, border_width=border_width,
                                 border_color=border_color)
        self.card.place(relx=0, rely=0, relwidth=1, relheight=1)


class BikeSharingFormPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller

        self.entries = {}
        self.model_ready = False
        self.df = None
        self.bundle = None
        self.meta = None
        self.metrics = {}
        self.best_name = ""
        self._bootstrapped = False

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main_content()

    def tkraise(self, *args, **kwargs):
        super().tkraise(*args, **kwargs)
        if not self._bootstrapped:
            self._bootstrapped = True
            self._check_and_train_bg()

    def update_theme(self):
        mode = ctk.get_appearance_mode()
        bg_col = CARD_BG[1] if mode == "Dark" else CARD_BG[0]
        fg_col = TEXT_MAIN[1] if mode == "Dark" else TEXT_MAIN[0]
        if hasattr(self, 'info_box'):
            self.info_box.configure(bg=bg_col, fg=fg_col)

        bg_col_tree = BG_MAIN[1] if mode == "Dark" else BG_MAIN[0]
        fg_col_tree = TEXT_MAIN[1] if mode == "Dark" else TEXT_MAIN[0]
        card_bg_tree = CARD_BG[1] if mode == "Dark" else CARD_BG[0]
        purple_hex = PURPLE[1] if mode == "Dark" else PURPLE[0]

        if hasattr(self, 'tree'):
            style = ttk.Style()
            style.theme_use("clam")
            style.configure("Treeview.Heading", font=("Cascadia Code", 11, "bold"), background=bg_col_tree,
                            foreground=fg_col_tree, borderwidth=0)
            style.configure("Treeview", font=("Cascadia Code", 10), rowheight=24, background=card_bg_tree,
                            fieldbackground=card_bg_tree, foreground=fg_col_tree, borderwidth=0)
            style.map("Treeview", background=[("selected", purple_hex)])
            self.tree.tag_configure("best", background=bg_col_tree if mode == "Dark" else "#F5F3FF",
                                    foreground=PURPLE_DARK[1] if mode == "Dark" else PURPLE_DARK[0])

    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self, fg_color=CARD_BG, corner_radius=20, width=210, border_width=1,
                               border_color=BORDER_COLOR)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        sidebar.grid_propagate(False)

        ctk.CTkLabel(sidebar, text="📊 EDA TOOLS", font=ctk.CTkFont(family="Cascadia Code", size=14, weight="bold"),
                     text_color=PURPLE).pack(pady=(20, 10), anchor="w", padx=15)

        eda_buttons = [
            ("🔥 Heatmap", self.show_heatmap),
            ("📈 Pred vs Actual", self.show_pred_actual),
            ("📊 Distribution", self.show_dist),
            ("🕐 Time Pattern", self.show_pattern),
        ]

        for text, cmd in eda_buttons:
            ctk.CTkButton(sidebar, text=text, fg_color="transparent", border_width=1, border_color=BORDER_COLOR,
                          text_color=TEXT_MAIN, hover_color=BORDER_COLOR,
                          font=ctk.CTkFont(family="Cascadia Code", size=11), corner_radius=10, height=32,
                          command=cmd).pack(pady=3, padx=10, fill="x")

        ctk.CTkLabel(sidebar, text="ℹ️ ABOUT & RANGES",
                     font=ctk.CTkFont(family="Cascadia Code", size=13, weight="bold"), text_color=PURPLE_DARK).pack(
            pady=(20, 5), anchor="w", padx=15)

        self.info_box = tk.Text(sidebar, font=("Cascadia Code", 10),
                                wrap="word", relief="flat", borderwidth=0, padx=8, pady=5, width=30, height=40)
        self.info_box.pack(anchor="w", padx=5, pady=(0, 15), fill="both", expand=True)
        self.info_box.tag_configure("green", foreground=GREEN[1], font=("Cascadia Code", 10, "bold"))

        self.info_box.insert("end", "Predict hourly bike rentals based on weather & season.\n")
        self.info_box.insert("end", "────────────────────────\n\n")
        self.info_box.insert("end", "INPUT FIELDS & RANGES:\n\n", "green")

        self.info_box.insert("end", "Season: 1 (Spring) to 4 (Winter)\n", "green")
        self.info_box.insert("end", "Year: 0 (2011) or 1 (2012)\n", "green")
        self.info_box.insert("end", "Month: 1 to 12\n", "green")
        self.info_box.insert("end", "Hour: 0 to 23\n", "green")
        self.info_box.insert("end", "Holiday: 0 (No), 1 (Yes)\n", "green")
        self.info_box.insert("end", "Weekday: 0 (Sun) to 6 (Sat)\n", "green")
        self.info_box.insert("end", "Working Day: 0 (No), 1 (Yes)\n", "green")
        self.info_box.insert("end", "Weather: 1 (Clear) to 4 (Heavy Rain)\n", "green")
        self.info_box.insert("end", "Temp, Atemp, Hum, Windspeed: 0.0 to 1.0\n\n", "green")

        self.info_box.insert("end", "TARGET:\n", "green")
        self.info_box.insert("end", "Hourly Rental Count (cnt)\n")
        self.info_box.configure(state="disabled")
        self.update_theme()

    def _build_main_content(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=0, minsize=160)
        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=1)

        # --- Left Column: Form ---
        form_card = ShadowCard(main, corner_radius=20, shadow_size=4)
        form_card.grid(row=0, column=0, padx=(0, 6), pady=(0, 6), sticky="nsew")

        ctk.CTkLabel(form_card.card, text="🚲 Bike Rental Prediction",
                     font=ctk.CTkFont(family="Cascadia Code", size=16, weight="bold"), text_color=TEXT_MAIN).pack(
            pady=(12, 6), anchor="w", padx=15)

        form_scroll = ctk.CTkScrollableFrame(form_card.card, fg_color="transparent")
        form_scroll.pack(padx=15, pady=(0, 5), fill="both", expand=True)
        form_scroll.grid_columnconfigure(0, weight=1)
        form_scroll.grid_columnconfigure(1, weight=1)

        fields = [
            ("season", "Season (1-4)"), ("weathersit", "Weather (1-4)"),
            ("mnth", "Month (1-12)"), ("hr", "Hour (0-23)"),
            ("weekday", "Weekday (0-6)"), ("yr", "Year (0/1)"),
            ("temp", "Temp (0-1)"), ("atemp", "Atemp (0-1)"),
            ("hum", "Humidity (0-1)"), ("windspeed", "Windspeed (0-1)"),
        ]

        row = 0
        for i in range(0, len(fields), 2):
            self._create_field(form_scroll, row, 0, fields[i])
            if i + 1 < len(fields): self._create_field(form_scroll, row, 1, fields[i + 1])
            row += 1

        # Combos
        self._create_combo(form_scroll, row, 0, "holiday", "Holiday", ["0", "1"])
        self._create_combo(form_scroll, row, 1, "workingday", "Working Day", ["0", "1"])

        # --- Buttons ---
        btn_frame_form = ctk.CTkFrame(form_card.card, fg_color="transparent")
        btn_frame_form.pack(fill="x", padx=15, pady=(0, 15))

        ctk.CTkButton(
            btn_frame_form, text="🎲 Random Fill",
            font=ctk.CTkFont(family="Cascadia Code", size=12, weight="bold"),
            fg_color="transparent", border_width=1, border_color=GREEN, text_color=GREEN,
            hover_color=BORDER_COLOR, corner_radius=10, height=32, command=self.fill_random
        ).pack(side="left", fill="x", expand=True, padx=(0, 5))

        ctk.CTkButton(
            btn_frame_form, text="🗑️ Clear",
            font=ctk.CTkFont(family="Cascadia Code", size=12),
            fg_color="transparent", border_width=1, border_color=RED, text_color=RED,
            hover_color=BORDER_COLOR, corner_radius=10, height=32, command=self.clear_form
        ).pack(side="left", fill="x", expand=True)

        # --- Right Column: Result ---
        result_card = ShadowCard(main, corner_radius=20, shadow_size=4)
        result_card.grid(row=0, column=1, padx=(6, 0), pady=(0, 6), sticky="nsew")

        ctk.CTkLabel(result_card.card, text="🔮 Result",
                     font=ctk.CTkFont(family="Cascadia Code", size=16, weight="bold"), text_color=TEXT_MAIN).pack(
            pady=(12, 6), anchor="w", padx=15)

        inner_bg = ("#F5F3FF", "#0D0B1F")
        self.result_inner = ctk.CTkFrame(result_card.card, fg_color=inner_bg, corner_radius=14, border_width=1,
                                         border_color=PURPLE)
        self.result_inner.pack(fill="both", expand=True, padx=15, pady=(5, 10))

        ctk.CTkLabel(self.result_inner, text="Predicted Rental Count",
                     font=ctk.CTkFont(family="Cascadia Code", size=12),
                     text_color=TEXT_MUTED).pack(pady=(20, 0))

        self.result_label = ctk.CTkLabel(
            self.result_inner, text="--",
            font=ctk.CTkFont(family="Cascadia Code", size=40, weight="bold"),
            text_color=PURPLE_DARK
        )
        self.result_label.pack(pady=(0, 5))

        self.unit_label = ctk.CTkLabel(self.result_inner, text="",
                                       font=ctk.CTkFont(family="Cascadia Code", size=13, weight="bold"),
                                       text_color=PURPLE)
        self.unit_label.pack(pady=(0, 10))

        self.status_label = ctk.CTkLabel(result_card.card, text="⏳ Training in background...",
                                         font=ctk.CTkFont(family="Cascadia Code", size=11), text_color=PURPLE)
        self.status_label.pack(pady=(0, 8))

        btn_frame_result = ctk.CTkFrame(result_card.card, fg_color="transparent")
        btn_frame_result.pack(fill="x", padx=15, pady=(0, 15))

        self.predict_btn = ctk.CTkButton(btn_frame_result, text="✨ Predict",
                                         font=ctk.CTkFont(family="Cascadia Code", size=14, weight="bold"),
                                         fg_color=PURPLE, hover_color=PURPLE_HOVER, text_color="#FFFFFF",
                                         corner_radius=12, height=40, command=self.predict, state="disabled")
        self.predict_btn.pack(fill="x")

        # --- Table ---
        table_card = ShadowCard(main, corner_radius=20, shadow_size=4)
        table_card.grid(row=1, column=0, columnspan=2, padx=0, pady=(6, 0), sticky="nsew")
        ctk.CTkLabel(table_card.card, text="📈 Model Comparison",
                     font=ctk.CTkFont(family="Cascadia Code", size=14, weight="bold"), text_color=TEXT_MAIN).pack(
            pady=(8, 2), anchor="w", padx=15)

        self.tree = ttk.Treeview(table_card.card, columns=("name", "rmse", "mae", "r2"), show="headings", height=3)
        self.tree.heading("name", text="Model")
        self.tree.heading("rmse", text="RMSE")
        self.tree.heading("mae", text="MAE")
        self.tree.heading("r2", text="R² Score")
        self.tree.column("name", width=240, anchor="center")
        self.tree.column("rmse", width=100, anchor="center")
        self.tree.column("mae", width=100, anchor="center")
        self.tree.column("r2", width=100, anchor="center")
        self.tree.pack(fill="x", padx=15, pady=(0, 8))

        self.update_theme()

    def _create_field(self, parent, row, col, field_data):
        key, label = field_data
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.grid(row=row, column=col, padx=5, pady=2, sticky="ew")
        ctk.CTkLabel(f, text=label, font=ctk.CTkFont(family="Cascadia Code", size=11, weight="bold"),
                     text_color=TEXT_MAIN).pack(anchor="w", pady=(0, 3))
        var = ctk.StringVar(value="")
        ctk.CTkEntry(f, textvariable=var, height=28, corner_radius=10, border_color=BORDER_COLOR, fg_color=BG_MAIN,
                     text_color=TEXT_MAIN, font=ctk.CTkFont(family="Cascadia Code", size=12)).pack(fill="x")
        self.entries[key] = var

    def _create_combo(self, parent, row, col, key, label, values):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.grid(row=row, column=col, padx=5, pady=2, sticky="ew")
        ctk.CTkLabel(f, text=label, font=ctk.CTkFont(family="Cascadia Code", size=11, weight="bold"),
                     text_color=TEXT_MAIN).pack(anchor="w", pady=(0, 3))
        var = ctk.StringVar(value=values[0])
        combo = ctk.CTkComboBox(f, values=values, variable=var, height=28, corner_radius=10, border_color=BORDER_COLOR,
                                fg_color=BG_MAIN, text_color=TEXT_MAIN,
                                font=ctk.CTkFont(family="Cascadia Code", size=12))
        combo.pack(fill="x")
        self.entries[key] = var

    def _show_chart(self, fig):
        fig.tight_layout()
        path = os.path.join(tempfile.gettempdir(), f"bike_eda_{random.randint(1000, 9999)}.png")
        fig.savefig(path, bbox_inches="tight");
        plt.close(fig)

        # Show in a new popup window
        win = ctk.CTkToplevel(self)
        win.title("EDA Chart")
        win.geometry("700x500")
        win.attributes("-topmost", True)
        img = ctk.CTkImage(light_image=Image.open(path), dark_image=Image.open(path), size=(650, 450))
        lbl = ctk.CTkLabel(win, text="", image=img)
        lbl.pack(expand=True, fill="both", padx=10, pady=10)
        lbl._img = img

    def show_heatmap(self):
        if self.df is None: return messagebox.showinfo("Wait", "Model is not ready yet.")
        fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=110)
        fig.patch.set_facecolor("#14141C");
        ax.set_facecolor("#1B1B25")
        ax.tick_params(colors="#C9C9D6", labelsize=8);
        ax.title.set_color("#F5F5F7")
        corr = self.df[pipeline.FEATURES + [pipeline.TARGET]].corr()
        im = ax.imshow(corr, cmap="magma", aspect="auto")
        ax.set_xticks(range(len(corr.columns)));
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=75, fontsize=7)
        ax.set_yticklabels(corr.columns, fontsize=7)
        ax.set_title("Correlation Heatmap")
        fig.colorbar(im, ax=ax, fraction=0.04)
        self._show_chart(fig)

    def show_pred_actual(self):
        if self.bundle is None: return messagebox.showinfo("Wait", "Model is not ready yet.")
        fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=110)
        fig.patch.set_facecolor("#14141C");
        ax.set_facecolor("#1B1B25")
        ax.tick_params(colors="#C9C9D6", labelsize=8);
        ax.title.set_color("#F5F5F7")
        y_true = np.asarray(self.bundle["test_true"]);
        y_pred = np.asarray(self.bundle["test_pred"])
        ax.scatter(y_true, y_pred, s=6, alpha=0.35, color="#8B5CF6")
        lim = [0, max(y_true.max(), y_pred.max()) * 1.05]
        ax.plot(lim, lim, "--", color="#10B981", linewidth=1)
        ax.set_xlabel("Actual cnt");
        ax.set_ylabel("Predicted cnt")
        ax.set_title(f"Predicted vs Actual ({LABEL_MAP[self.best_name]})")
        self._show_chart(fig)

    def show_dist(self):
        if self.df is None: return messagebox.showinfo("Wait", "Model is not ready yet.")
        fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=110)
        fig.patch.set_facecolor("#14141C");
        ax.set_facecolor("#1B1B25")
        ax.tick_params(colors="#C9C9D6", labelsize=8);
        ax.title.set_color("#F5F5F7")
        ax.hist(self.df[pipeline.TARGET], bins=50, color="#7C3AED", alpha=0.85)
        ax.axvline(self.df[pipeline.TARGET].mean(), color="#10B981", linestyle="--", linewidth=1.2,
                   label=f"mean = {self.df[pipeline.TARGET].mean():.0f}")
        ax.set_xlabel("cnt");
        ax.set_ylabel("Frequency");
        ax.set_title("Target Distribution")
        ax.legend(facecolor="#1B1B25", edgecolor="#3A3A4A", labelcolor="#C9C9D6", fontsize=8)
        self._show_chart(fig)

    def show_pattern(self):
        if self.df is None: return messagebox.showinfo("Wait", "Model is not ready yet.")
        fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=110)
        fig.patch.set_facecolor("#14141C");
        ax.set_facecolor("#1B1B25")
        ax.tick_params(colors="#C9C9D6", labelsize=8);
        ax.title.set_color("#F5F5F7")
        hourly = self.df.groupby("hr")[pipeline.TARGET].mean();
        monthly = self.df.groupby("mnth")[pipeline.TARGET].mean()
        ax.plot(hourly.index, hourly.values, marker="o", color="#8B5CF6", linewidth=1.8, label="Hourly avg")
        ax2 = ax.twinx();
        ax2.plot(monthly.index, monthly.values, marker="s", color="#10B981", linewidth=1.5, label="Monthly avg")
        ax2.tick_params(colors="#C9C9D6", labelsize=8)
        ax.set_xlabel("Hour / Month");
        ax.set_ylabel("Avg cnt (hourly)", color="#8B5CF6");
        ax2.set_ylabel("Avg cnt (monthly)", color="#10B981")
        ax.set_title("Hourly & Monthly Rental Pattern")
        h1, l1 = ax.get_legend_handles_labels();
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", facecolor="#1B1B25", edgecolor="#3A3A4A", labelcolor="#C9C9D6",
                  fontsize=8)
        self._show_chart(fig)

    # ================= LOGIC =================
    def clear_form(self):
        for var in self.entries.values(): var.set("")
        self.entries["holiday"].set("0")
        self.entries["workingday"].set("0")
        self.result_label.configure(text="--")
        self.unit_label.configure(text="")

    def fill_random(self):
        self.entries["season"].set(str(random.randint(1, 4)))
        self.entries["yr"].set(str(random.randint(0, 1)))
        self.entries["mnth"].set(str(random.randint(1, 12)))
        self.entries["hr"].set(str(random.randint(0, 23)))
        self.entries["holiday"].set(str(random.randint(0, 1)))
        self.entries["weekday"].set(str(random.randint(0, 6)))
        self.entries["workingday"].set(str(random.randint(0, 1)))
        self.entries["weathersit"].set(str(random.randint(1, 4)))
        self.entries["temp"].set(f"{random.uniform(0.1, 0.9):.2f}")
        self.entries["atemp"].set(f"{random.uniform(0.1, 0.9):.2f}")
        self.entries["hum"].set(f"{random.uniform(0.2, 0.9):.2f}")
        self.entries["windspeed"].set(f"{random.uniform(0.0, 0.5):.2f}")

    def _check_and_train_bg(self):
        data = pipeline.load_bundle()
        if data:
            self._on_train_success(data[1])
        else:
            self.status_label.configure(text="⏳ Training in background...", text_color=PURPLE)
            threading.Thread(target=self._train_bg, daemon=True).start()

    def _train_bg(self):
        try:
            pipeline.train_and_evaluate()
            data = pipeline.load_bundle()
            self.after(0, lambda: self._on_train_success(data[1] if data else {}))
        except Exception as exc:
            err_msg = str(exc)
            self.after(0, lambda: self._on_train_error(err_msg))

    def _on_train_success(self, meta):
        self._load_model()
        self.status_label.configure(text=f"✅ Ready — best: {LABEL_MAP[self.best_name]}", text_color=GREEN)

    def _on_train_error(self, msg):
        self.model_ready = False
        self.predict_btn.configure(state="disabled")
        self.status_label.configure(text=f"❌ Training failed: {msg[:40]}", text_color=RED)
        messagebox.showerror("Training Error", msg)

    def _load_model(self):
        data = pipeline.load_bundle()
        if not data: return
        self.bundle, self.meta = data
        self.metrics = self.meta.get("metrics", {})
        self.best_name = self.meta.get("best_model", "")
        self.df = pipeline.load_data()
        self.model_ready = True
        self.predict_btn.configure(state="normal")
        self._show_metrics()

    def _show_metrics(self):
        for item in self.tree.get_children(): self.tree.delete(item)
        for name, m in self.metrics.items():
            tags = ("best",) if name == self.best_name else ()
            display = f"⭐ {LABEL_MAP[name]}" if name == self.best_name else LABEL_MAP[name]
            rmse = f"{m['RMSE']:.2f}" if m.get("RMSE") is not None else "—"
            mae = f"{m['MAE']:.2f}" if m.get("MAE") is not None else "—"
            r2 = f"{m['R2']:.4f}" if m.get("R2") is not None else "—"
            self.tree.insert("", "end", values=(display, rmse, mae, r2), tags=tags)

    def predict(self):
        if not self.model_ready:
            messagebox.showwarning("Not Ready", "Model is not ready.")
            return
        try:
            row_dict = {}
            for k, var in self.entries.items():
                v = var.get().strip()
                if v == "": raise ValueError(f"Field {k} is empty.")
                row_dict[k] = float(v) if k not in ["holiday", "workingday"] else int(v)

            pred, label = pipeline.predict_row(self.bundle, self.meta, row_dict)

            color_map = {"Low": GREEN, "Moderate": AMBER, "High": ORANGE, "Very High": RED}
            self.result_label.configure(text=f"{pred:,.0f}", text_color=PURPLE_DARK)
            self.unit_label.configure(text=f"Demand level: {label}", text_color=color_map[label])
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error:\n{e}")