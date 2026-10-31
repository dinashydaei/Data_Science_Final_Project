import os
import threading
import traceback
import tkinter as tk
from tkinter import messagebox, ttk
import customtkinter as ctk
import joblib

from pipelines import car_evaluation_classification as pipeline
from utils import visuals

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


class ShadowCard(ctk.CTkFrame):
    def __init__(self, parent, corner_radius=20, shadow_size=4, fg_color=CARD_BG, border_width=1,
                 border_color=BORDER_COLOR, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.shadow = ctk.CTkFrame(self, fg_color=CARD_SHADOW, corner_radius=corner_radius)
        self.shadow.place(relx=0, rely=0, relwidth=1, relheight=1, x=shadow_size, y=shadow_size)
        self.card = ctk.CTkFrame(self, fg_color=fg_color, corner_radius=corner_radius,
                                 border_width=border_width, border_color=border_color)
        self.card.place(relx=0, rely=0, relwidth=1, relheight=1)


class CarFormPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller

        self.entries = {}
        self.model_ready = False
        self.df = None
        self.bundle = None
        self.metrics = {}
        self.best_name = ""

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main_content()
        self._check_and_train_bg()

    # ------------------------------------------------------------
    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self, fg_color=CARD_BG, corner_radius=20, width=210,
                               border_width=1, border_color=BORDER_COLOR)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        sidebar.grid_propagate(False)

        ctk.CTkLabel(sidebar, text="📊 EDA TOOLS",
                     font=ctk.CTkFont(family="Cascadia Code", size=14, weight="bold"),
                     text_color=PURPLE).pack(pady=(20, 10), anchor="w", padx=15)

        eda_buttons = [
            ("🔥 Heatmap", self.show_heatmap),
            ("🎯 Confusion Matrix", self.show_confusion_matrix),
        ]
        for text, cmd in eda_buttons:
            ctk.CTkButton(sidebar, text=text, fg_color="transparent", border_width=1,
                          border_color=BORDER_COLOR, text_color=TEXT_MAIN, hover_color=BORDER_COLOR,
                          font=ctk.CTkFont(family="Cascadia Code", size=11),
                          corner_radius=10, height=32, command=cmd
                          ).pack(pady=3, padx=10, fill="x")

        ctk.CTkLabel(sidebar, text="ℹ️ ABOUT & OPTIONS",
                     font=ctk.CTkFont(family="Cascadia Code", size=13, weight="bold"),
                     text_color=PURPLE_DARK).pack(pady=(20, 5), anchor="w", padx=15)

        self.info_box = tk.Text(sidebar, font=("Cascadia Code", 10), wrap="word",
                                relief="flat", borderwidth=0, padx=8, pady=5, width=30, height=40)
        self.info_box.pack(anchor="w", padx=5, pady=(0, 15), fill="both", expand=True)
        self.info_box.tag_configure("green", foreground=GREEN[1],
                                    font=("Cascadia Code", 10, "bold"))

        self.info_box.insert("end", "Predict car acceptability from attributes.\n")
        self.info_box.insert("end", "────────────────────────\n\n")
        self.info_box.insert("end", "BUYING PRICE:\n", "green")
        self.info_box.insert("end", "vhigh, high, med, low\n\n")
        self.info_box.insert("end", "MAINTENANCE:\n", "green")
        self.info_box.insert("end", "vhigh, high, med, low\n\n")
        self.info_box.insert("end", "DOORS:\n", "green")
        self.info_box.insert("end", "2, 3, 4, 5more\n\n")
        self.info_box.insert("end", "PERSONS:\n", "green")
        self.info_box.insert("end", "2, 4, more\n\n")
        self.info_box.insert("end", "LUGGAGE BOOT:\n", "green")
        self.info_box.insert("end", "small, med, big\n\n")
        self.info_box.insert("end", "SAFETY:\n", "green")
        self.info_box.insert("end", "low, med, high\n\n")
        self.info_box.insert("end", "TARGET:\n", "green")
        self.info_box.insert("end", "Car acceptability (unacc / acc / good / vgood)")
        self.info_box.configure(state="disabled")

    # ------------------------------------------------------------
    def _build_main_content(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=0, minsize=160)
        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=1)

        # ---------- فرم ----------
        form_card = ShadowCard(main)
        form_card.grid(row=0, column=0, padx=(0, 6), pady=(0, 6), sticky="nsew")

        ctk.CTkLabel(form_card.card, text="🚗 Car Features",
                     font=ctk.CTkFont(family="Cascadia Code", size=16, weight="bold"),
                     text_color=TEXT_MAIN).pack(pady=(12, 6), anchor="w", padx=15)

        form_scroll = ctk.CTkScrollableFrame(form_card.card, fg_color="transparent")
        form_scroll.pack(padx=15, pady=(0, 5), fill="both", expand=True)
        form_scroll.grid_columnconfigure(0, weight=1)
        form_scroll.grid_columnconfigure(1, weight=1)

        fields = [
            ("buying", "Buying Price", ["vhigh", "high", "med", "low"]),
            ("maint", "Maintenance", ["vhigh", "high", "med", "low"]),
            ("doors", "Doors", ["2", "3", "4", "5more"]),
            ("persons", "Persons", ["2", "4", "more"]),
            ("lug_boot", "Luggage Boot", ["small", "med", "big"]),
            ("safety", "Safety", ["low", "med", "high"]),
        ]

        row = 0
        for i in range(0, len(fields), 2):
            self._create_combo(form_scroll, row, 0, fields[i])
            if i + 1 < len(fields):
                self._create_combo(form_scroll, row, 1, fields[i + 1])
            row += 1

        # دکمه‌ها
        btn_frame = ctk.CTkFrame(form_card.card, fg_color="transparent")
        btn_frame.pack(fill="x", padx=15, pady=(0, 15))

        ctk.CTkButton(
            btn_frame, text="🗑️ Clear",
            font=ctk.CTkFont(family="Cascadia Code", size=12),
            fg_color="transparent", border_width=1, border_color=RED,
            text_color=RED, hover_color=BORDER_COLOR, corner_radius=10, height=32,
            command=self.clear_form
        ).pack(fill="x")

        # ---------- نتیجه ----------
        result_card = ShadowCard(main)
        result_card.grid(row=0, column=1, padx=(6, 0), pady=(0, 6), sticky="nsew")

        ctk.CTkLabel(result_card.card, text="🔮 Result",
                     font=ctk.CTkFont(family="Cascadia Code", size=16, weight="bold"),
                     text_color=TEXT_MAIN).pack(pady=(12, 6), anchor="w", padx=15)

        inner_bg = ("#F5F3FF", "#0D0B1F")
        self.result_inner = ctk.CTkFrame(result_card.card, fg_color=inner_bg, corner_radius=14,
                                         border_width=1, border_color=PURPLE)
        self.result_inner.pack(fill="both", expand=True, padx=15, pady=(5, 10))

        ctk.CTkLabel(self.result_inner, text="Car Acceptability",
                     font=ctk.CTkFont(family="Cascadia Code", size=12),
                     text_color=TEXT_MUTED).pack(pady=(20, 0))

        self.result_label = ctk.CTkLabel(self.result_inner, text="--",
                                         font=ctk.CTkFont(family="Cascadia Code", size=22, weight="bold"),
                                         text_color=PURPLE_DARK, wraplength=240)
        self.result_label.pack(pady=(0, 5))

        self.conf_label = ctk.CTkLabel(self.result_inner, text="",
                                       font=ctk.CTkFont(family="Cascadia Code", size=13, weight="bold"),
                                       text_color=PURPLE)
        self.conf_label.pack(pady=(0, 10))

        self.status_label = ctk.CTkLabel(result_card.card, text="⏳ Training in background...",
                                         font=ctk.CTkFont(family="Cascadia Code", size=11),
                                         text_color=PURPLE)
        self.status_label.pack(pady=(0, 8))

        btn_result = ctk.CTkFrame(result_card.card, fg_color="transparent")
        btn_result.pack(fill="x", padx=15, pady=(0, 15))

        self.predict_btn = ctk.CTkButton(btn_result, text="✨ Predict",
                                         font=ctk.CTkFont(family="Cascadia Code", size=14, weight="bold"),
                                         fg_color=PURPLE, hover_color=PURPLE_HOVER, text_color="#FFFFFF",
                                         corner_radius=12, height=40, command=self.predict, state="disabled")
        self.predict_btn.pack(fill="x")

        # ---------- جدول مقایسه ----------
        table_card = ShadowCard(main)
        table_card.grid(row=1, column=0, columnspan=2, padx=0, pady=(6, 0), sticky="nsew")

        ctk.CTkLabel(table_card.card, text="📈 Model Comparison",
                     font=ctk.CTkFont(family="Cascadia Code", size=14, weight="bold"),
                     text_color=TEXT_MAIN).pack(pady=(8, 2), anchor="w", padx=15)

        self.tree = ttk.Treeview(table_card.card,
                                 columns=("name", "acc", "f1"),
                                 show="headings", height=4)
        self.tree.heading("name", text="Model")
        self.tree.heading("acc", text="Accuracy")
        self.tree.heading("f1", text="F1-Score")
        self.tree.column("name", width=220, anchor="center")
        self.tree.column("acc", width=100, anchor="center")
        self.tree.column("f1", width=100, anchor="center")
        self.tree.pack(fill="x", padx=15, pady=(0, 8))

        self.update_theme()

    # ------------------------------------------------------------
    def _create_combo(self, parent, row, col, field_data):
        key, label, values = field_data
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.grid(row=row, column=col, padx=5, pady=2, sticky="ew")
        ctk.CTkLabel(f, text=label,
                     font=ctk.CTkFont(family="Cascadia Code", size=11, weight="bold"),
                     text_color=TEXT_MAIN).pack(anchor="w", pady=(0, 3))
        var = ctk.StringVar(value=values[0])
        ctk.CTkComboBox(f, values=values, variable=var, height=28, corner_radius=10,
                        border_color=BORDER_COLOR, fg_color=BG_MAIN, text_color=TEXT_MAIN,
                        font=ctk.CTkFont(family="Cascadia Code", size=12)).pack(fill="x")
        self.entries[key] = var

    # ------------------------------------------------------------
    def update_theme(self):
        mode = ctk.get_appearance_mode()
        bg_col = CARD_BG[1] if mode == "Dark" else CARD_BG[0]
        fg_col = TEXT_MAIN[1] if mode == "Dark" else TEXT_MAIN[0]
        self.info_box.configure(bg=bg_col, fg=fg_col)

        bg_col_tree = BG_MAIN[1] if mode == "Dark" else BG_MAIN[0]
        card_bg_tree = CARD_BG[1] if mode == "Dark" else CARD_BG[0]
        purple_hex = PURPLE[1] if mode == "Dark" else PURPLE[0]

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview.Heading", font=("Cascadia Code", 10, "bold"),
                        background=bg_col_tree, foreground=fg_col, borderwidth=0)
        style.configure("Treeview", font=("Cascadia Code", 10), rowheight=24,
                        background=card_bg_tree, fieldbackground=card_bg_tree,
                        foreground=fg_col, borderwidth=0)
        style.map("Treeview", background=[("selected", purple_hex)])
        self.tree.tag_configure("best",
                                background=bg_col_tree if mode == "Dark" else "#F5F3FF",
                                foreground=PURPLE_DARK[1] if mode == "Dark" else PURPLE_DARK[0])

    # ------------------------------------------------------------
    def show_heatmap(self):
        if self.df is not None:
            self.after(0, lambda: visuals.plot_heatmap(self.df))

    def show_confusion_matrix(self):
        if self.bundle is not None and "y_test" in self.bundle:
            self.after(0, lambda: visuals.plot_confusion_matrix(
                self.bundle["y_test"], self.bundle["y_pred"], self.bundle["classes"]
            ))

    # ------------------------------------------------------------
    def clear_form(self):
        for key, var in self.entries.items():
            try:
                var.set(var._value[0])
            except Exception:
                var.set("")
        self.result_label.configure(text="--")
        self.conf_label.configure(text="")

    # ------------------------------------------------------------
    def _check_and_train_bg(self):
        if os.path.exists(pipeline.MODEL_PATH):
            try:
                self._load_model()
                return
            except Exception:
                pass
        threading.Thread(target=self._train_bg, daemon=True).start()

    def _train_bg(self):
        try:
            results, best = pipeline.train_and_evaluate()
            try:
                self.after(0, lambda: self._on_train_success(results, best))
            except RuntimeError:
                pass
        except Exception as exc:
            err_msg = str(exc)
            tb = traceback.format_exc()
            try:
                self.after(0, lambda m=err_msg, t=tb: self._on_train_error(m, t))
            except RuntimeError:
                pass

    def _on_train_success(self, results, best):
        try:
            self._load_model()
            self.status_label.configure(text=f"✅ Ready — best: {best}", text_color=GREEN)
        except Exception as exc:
            self._on_train_error(str(exc), "")

    def _on_train_error(self, msg, tb=""):
        self.model_ready = False
        try:
            self.predict_btn.configure(state="disabled")
        except Exception:
            pass
        try:
            self.status_label.configure(text=f"❌ Training failed: {msg[:80]}", text_color=RED)
        except Exception:
            pass
        try:
            self.after(200, lambda m=f"{msg}\n\n{tb[-600:] if tb else ''}":
                       messagebox.showerror("Training Error", m))
        except Exception:
            pass

    def _load_model(self):
        self.bundle = joblib.load(pipeline.MODEL_PATH)
        self.metrics = self.bundle.get("metrics", {})
        self.best_name = self.bundle.get("best_name", "")
        self.df = pipeline.load_data()
        self.model_ready = True
        self.status_label.configure(text=f"✅ Ready — best: {self.best_name}", text_color=GREEN)
        self.predict_btn.configure(state="normal")
        self._show_metrics()

    def _show_metrics(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for name, m in self.metrics.items():
            tags = ("best",) if name == self.best_name else ()
            display = f"⭐ {name}" if name == self.best_name else name
            self.tree.insert("", "end", values=(
                display,
                f"{m['accuracy'] * 100:.2f}%",
                f"{m['f1']:.4f}",
            ), tags=tags)

    # ------------------------------------------------------------
    def predict(self):
        if not self.model_ready:
            messagebox.showwarning("Not Ready", "مدل هنوز آماده نیست.")
            return
        try:
            vals = {k: v.get().strip() for k, v in self.entries.items()}
            res = pipeline.predict(**vals)
            self.result_label.configure(text=f"{res['label']}")
            self.conf_label.configure(text="")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"خطا در پیش‌بینی:\n{e}")