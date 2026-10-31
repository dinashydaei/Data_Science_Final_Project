import os
import tkinter as tk
from tkinter import messagebox, ttk
import threading
import customtkinter as ctk
import joblib
from pipelines import daily_demand_regression as pipeline
from utils import visuals

BG_DARK = ("#F3F4F6", "#0A0A10")
CARD_BG = ("#FFFFFF", "#151520")
BORDER_COLOR = ("#E5E7EB", "#2A2A35")
PURPLE = ("#7C3AED", "#8B5CF6")
PURPLE_HOVER = ("#6D28D9", "#7C3AED")
GREEN = ("#059669", "#10B981")
TEXT_MAIN = ("#111827", "#FFFFFF")
TEXT_MUTED = ("#6B7280", "#9CA3AF")


class DailyDemandFormPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller
        self.entries = {}
        self.model_ready = False
        self.df = None
        self.bundle = None

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main_content()
        self._check_and_train_bg()

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, fg_color=CARD_BG, corner_radius=15, width=240, border_width=1,
                                    border_color=BORDER_COLOR)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.sidebar.grid_propagate(False)

        ctk.CTkLabel(self.sidebar, text="📊 EDA TOOLS", font=ctk.CTkFont(family="Cascadia Code", size=14, weight="bold"),
                     text_color=GREEN).pack(pady=(20, 10), anchor="w", padx=15)

        ctk.CTkButton(self.sidebar, text="🔥 Heatmap", fg_color="transparent", border_width=1, border_color=BORDER_COLOR,
                      text_color=TEXT_MAIN, hover_color=BORDER_COLOR, font=ctk.CTkFont(family="Cascadia Code", size=12),
                      command=self.show_heatmap).pack(pady=3, padx=10, fill="x")

        ctk.CTkButton(self.sidebar, text="📈 Pairplot", fg_color="transparent", border_width=1,
                      border_color=BORDER_COLOR,
                      text_color=TEXT_MAIN, hover_color=BORDER_COLOR, font=ctk.CTkFont(family="Cascadia Code", size=12),
                      command=self.show_pairplot).pack(pady=3, padx=10, fill="x")

        # --- بخش توضیحات و رنج‌ها در سایدبار ---
        ctk.CTkLabel(self.sidebar, text="ℹ️ ABOUT & RANGES",
                     font=ctk.CTkFont(family="Cascadia Code", size=13, weight="bold"), text_color=PURPLE).pack(
            pady=(20, 5), anchor="w", padx=15)

        info_text = (
            "Predicts Total Daily Orders.\n\n"
            "• Week: 1 - 5\n"
            "• Day: 2 - 6\n"
            "• Non-Urgent: 0 - 300\n"
            "• Urgent: 0 - 200\n"
            "• Type A: 0 - 100\n"
            "• Type B: 0 - 200\n"
            "• Type C: 0 - 300\n"
            "• Fiscal: 0 - 100\n"
            "• Traffic: 10K - 60K\n"
            "• Bank 1: 10K - 150K\n"
            "• Bank 2: 5K - 40K\n"
            "• Bank 3: 10K - 100K"
        )

        info_box = ctk.CTkTextbox(self.sidebar, fg_color="transparent", text_color=TEXT_MUTED,
                                  font=ctk.CTkFont(family="Cascadia Code", size=11), wrap="word", height=300)
        info_box.pack(anchor="w", padx=15, fill="both", expand=True)
        info_box.insert("0.0", info_text)
        info_box.configure(state="disabled")  # غیرقابل ویرایش

    def _build_main_content(self):
        self.main_content = ctk.CTkFrame(self, fg_color="transparent")
        self.main_content.grid(row=0, column=1, sticky="nsew")

        self.main_content.grid_rowconfigure(0, weight=1)
        self.main_content.grid_rowconfigure(1, weight=1)
        self.main_content.grid_columnconfigure(0, weight=1)
        self.main_content.grid_columnconfigure(1, weight=1)

        # کارت فرم (سمت چپ)
        input_card = ctk.CTkFrame(self.main_content, fg_color=CARD_BG, corner_radius=15, border_width=1,
                                  border_color=BORDER_COLOR)
        input_card.grid(row=0, column=0, padx=(0, 5), pady=(0, 5), sticky="nsew")

        ctk.CTkLabel(input_card, text="📦 Order Parameters",
                     font=ctk.CTkFont(family="Cascadia Code", size=16, weight="bold"), text_color=TEXT_MAIN).pack(
            pady=(10, 5), anchor="w", padx=15)

        form_frame = ctk.CTkFrame(input_card, fg_color="transparent")
        form_frame.pack(padx=15, fill="both", expand=True)

        form_frame.grid_columnconfigure(0, weight=1)
        form_frame.grid_columnconfigure(1, weight=1)

        # فقط کلید و نام فیلد (بدون رنج)
        fields = [
            ("week_of_month", "Week of Month"),
            ("day_of_week", "Day of Week"),
            ("non_urgent", "Non-Urgent"),
            ("urgent", "Urgent"),
            ("order_a", "Type A Orders"),
            ("order_b", "Type B Orders"),
            ("order_c", "Type C Orders"),
            ("fiscal", "Fiscal Sector"),
            ("traffic", "Traffic Ctrl"),
            ("bank1", "Banking 1"),
            ("bank2", "Banking 2"),
            ("bank3", "Banking 3")
        ]

        row = 0
        for i in range(0, len(fields), 2):
            self._create_field(form_frame, row, 0, fields[i])
            if i + 1 < len(fields):
                self._create_field(form_frame, row, 1, fields[i + 1])
            row += 1

        # کارت نتیجه (سمت راست)
        self.result_card = ctk.CTkFrame(self.main_content, fg_color=CARD_BG, corner_radius=15, border_width=1,
                                        border_color=BORDER_COLOR)
        self.result_card.grid(row=0, column=1, padx=(5, 0), pady=(0, 5), sticky="nsew")

        ctk.CTkLabel(self.result_card, text="🔮 Result",
                     font=ctk.CTkFont(family="Cascadia Code", size=16, weight="bold"), text_color=TEXT_MAIN).pack(
            pady=(10, 5), anchor="w", padx=15)

        inner_bg = ("#ECFDF5", "#0D1F1A")
        self.result_inner = ctk.CTkFrame(self.result_card, fg_color=inner_bg, corner_radius=10, border_width=1,
                                         border_color=GREEN)
        self.result_inner.pack(fill="both", expand=True, padx=15, pady=(5, 10))

        ctk.CTkLabel(self.result_inner, text="Predicted Total Orders",
                     font=ctk.CTkFont(family="Cascadia Code", size=12), text_color=TEXT_MUTED).pack(pady=(20, 0))
        self.result_label = ctk.CTkLabel(self.result_inner, text="--",
                                         font=ctk.CTkFont(family="Cascadia Code", size=40, weight="bold"),
                                         text_color=GREEN)
        self.result_label.pack()
        self.status_label = ctk.CTkLabel(self.result_card, text="⏳ Training...",
                                         font=ctk.CTkFont(family="Cascadia Code", size=11), text_color=PURPLE)
        self.status_label.pack(pady=(0, 10))

        # دکمه‌ها زیر نتیجه
        btn_frame = ctk.CTkFrame(self.result_card, fg_color="transparent")
        btn_frame.pack(fill="x", padx=15, pady=(0, 15))

        self.predict_btn = ctk.CTkButton(btn_frame, text="✨ Predict",
                                         font=ctk.CTkFont(family="Cascadia Code", size=14, weight="bold"),
                                         fg_color=PURPLE, hover_color=PURPLE_HOVER, text_color="#FFFFFF",
                                         corner_radius=8, height=35, command=self.predict, state="disabled")
        self.predict_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        ctk.CTkButton(btn_frame, text="🗑️ Clear", font=ctk.CTkFont(family="Cascadia Code", size=14),
                      fg_color="transparent", border_width=1, border_color="#E74C3C", text_color="#E74C3C",
                      hover_color=BORDER_COLOR, height=35, corner_radius=8, command=self.clear_form).pack(side="left",
                                                                                                          fill="x",
                                                                                                          expand=True)

        # جدول مقایسه (پایین)
        self.table_card = ctk.CTkFrame(self.main_content, fg_color=CARD_BG, corner_radius=15, border_width=1,
                                       border_color=BORDER_COLOR)
        self.table_card.grid(row=1, column=0, columnspan=2, padx=0, pady=(5, 0), sticky="nsew")

        ctk.CTkLabel(self.table_card, text="📈 Model Comparison",
                     font=ctk.CTkFont(family="Cascadia Code", size=16, weight="bold"), text_color=TEXT_MAIN).pack(
            pady=(10, 5), anchor="w", padx=15)

        self.tree = ttk.Treeview(self.table_card, columns=("name", "R2", "RMSE", "MAE"), show="headings", height=4)
        self.tree.heading("name", text="Model");
        self.tree.heading("R2", text="R²");
        self.tree.heading("RMSE", text="RMSE");
        self.tree.heading("MAE", text="MAE")
        self.tree.column("name", width=200, anchor="center");
        self.tree.column("R2", width=100, anchor="center");
        self.tree.column("RMSE", width=100, anchor="center");
        self.tree.column("MAE", width=100, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        self.update_theme()

    def _create_field(self, parent, row, col, field_data):
        key, label = field_data
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.grid(row=row, column=col, padx=5, pady=5, sticky="ew")

        ctk.CTkLabel(f, text=label, font=ctk.CTkFont(family="Cascadia Code", size=11, weight="bold"),
                     text_color=TEXT_MAIN).pack(anchor="w", pady=(0, 2))

        var = ctk.StringVar(value="")
        entry = ctk.CTkEntry(f, textvariable=var, height=30, corner_radius=6,
                             border_color=BORDER_COLOR, fg_color=BG_DARK, text_color=TEXT_MAIN,
                             font=ctk.CTkFont(family="Cascadia Code", size=12))
        entry.pack(fill="x")
        self.entries[key] = var

    def update_theme(self):
        mode = ctk.get_appearance_mode()
        bg_col = BG_DARK[1] if mode == "Dark" else BG_DARK[0]
        fg_col = TEXT_MAIN[1] if mode == "Dark" else TEXT_MAIN[0]
        card_bg = CARD_BG[1] if mode == "Dark" else CARD_BG[0]
        green_hex = GREEN[1] if mode == "Dark" else GREEN[0]
        purple_hex = PURPLE[1] if mode == "Dark" else PURPLE[0]

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview.Heading", font=("Cascadia Code", 11, "bold"), background=bg_col, foreground=fg_col,
                        borderwidth=0)
        style.configure("Treeview", font=("Cascadia Code", 11), rowheight=25, background=card_bg,
                        fieldbackground=card_bg, foreground=fg_col, borderwidth=0)
        style.map("Treeview", background=[("selected", purple_hex)])
        self.tree.tag_configure("best", background=bg_col if mode == "Dark" else "#F3F4F6", foreground=green_hex)

    def show_heatmap(self):
        if self.df is not None: self.after(0, lambda: visuals.plot_heatmap(self.df))

    def show_pairplot(self):
        if self.df is not None: self.after(0, lambda: visuals.plot_pairplot(self.df, pipeline.TARGET))

    def clear_form(self):
        for var in self.entries.values():
            var.set("")
        self.result_label.configure(text="--")

    def _check_and_train_bg(self):
        if os.path.exists(pipeline.MODEL_PATH):
            self._load_model()
        else:
            threading.Thread(target=self._train_bg, daemon=True).start()

    def _train_bg(self):
        try:
            results, best = pipeline.train_and_evaluate()
            try:
                self.after(0, lambda: self._on_train_success(results, best))
            except RuntimeError:
                pass
        except Exception as e:
            err = str(e)
            try:
                self.after(0, lambda: self.status_label.configure(text=f"❌ {err}", text_color="#E74C3C"))
            except RuntimeError:
                pass

    def _load_model(self):
        self.bundle = joblib.load(pipeline.MODEL_PATH)
        self.metrics = self.bundle["metrics"]
        self.best_name = self.bundle["best_name"]
        self.model_ready = True
        self.df = pipeline.load_data()
        self.status_label.configure(text="✅ Ready", text_color=GREEN)
        self.predict_btn.configure(state="normal")
        self._show_metrics()

    def _on_train_success(self, results, best):
        self._load_model()

    def _show_metrics(self):
        for item in self.tree.get_children(): self.tree.delete(item)
        for name, m in self.metrics.items():
            tags = ("best",) if name == self.best_name else ()
            display_name = f"⭐ {name}" if name == self.best_name else name
            self.tree.insert("", "end", values=(display_name, f"{m['r2']:.4f}", f"{m['rmse']:.4f}", f"{m['mae']:.4f}"),
                             tags=tags)

    def predict(self):
        if not self.model_ready: return
        try:
            vals = {k: v.get().strip() for k, v in self.entries.items()}

            if any(v == "" for v in vals.values()):
                raise ValueError("Please fill all input fields before predicting.")

            numeric = {
                "week_of_month": int(vals["week_of_month"]), "day_of_week": int(vals["day_of_week"]),
                "non_urgent": float(vals["non_urgent"]), "urgent": float(vals["urgent"]),
                "order_a": float(vals["order_a"]), "order_b": float(vals["order_b"]), "order_c": float(vals["order_c"]),
                "fiscal": float(vals["fiscal"]), "traffic": float(vals["traffic"]),
                "bank1": float(vals["bank1"]), "bank2": float(vals["bank2"]), "bank3": float(vals["bank3"])
            }

            res = pipeline.predict(**numeric)
            self.result_label.configure(text=f"{res['total']:.2f}")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input:\n{e}")