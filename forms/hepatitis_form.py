import os
import traceback
import threading
import tkinter as tk
from tkinter import messagebox, ttk
import customtkinter as ctk
import joblib

from pipelines import hepatitis_classification as pipeline
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
        self.card = ctk.CTkFrame(self, fg_color=fg_color, corner_radius=corner_radius, border_width=border_width,
                                 border_color=border_color)
        self.card.place(relx=0, rely=0, relwidth=1, relheight=1)


class HepatitisFormPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller

        self.entries = {}
        self.combos = {}  # برای فیلدهای کشویی (1 یا 2)

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

    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self, fg_color=CARD_BG, corner_radius=20, width=210, border_width=1,
                               border_color=BORDER_COLOR)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        sidebar.grid_propagate(False)

        ctk.CTkLabel(sidebar, text="📊 EDA TOOLS", font=ctk.CTkFont(family="Cascadia Code", size=14, weight="bold"),
                     text_color=PURPLE).pack(pady=(20, 10), anchor="w", padx=15)

        eda_buttons = [
            ("🔥 Heatmap", self.show_heatmap),
            ("🎯 Plot", self.show_pairplot),
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

        self.info_box.insert("end", "Predict the chance of survival.\n")
        self.info_box.insert("end", "────────────────────────\n\n")
        self.info_box.insert("end", "INPUT FIELDS & RANGES:\n\n", "green")

        self.info_box.insert("end", "AGE: Range = 7 - 80\n", "green")
        self.info_box.insert("end", "Patient age in years.\n\n")

        self.info_box.insert("end", "SEX: 1 or 2\n", "green")
        self.info_box.insert("end", "1 = Male, 2 = Female.\n\n")

        self.info_box.insert("end", "STEROID: 1 or 2\n", "green")
        self.info_box.insert("end", "1 = No, 2 = Yes (steroid use).\n\n")

        self.info_box.insert("end", "ANTIVIRALS: 1 or 2\n", "green")
        self.info_box.insert("end", "1 = No, 2 = Yes.\n\n")

        self.info_box.insert("end", "FATIGUE: 1 or 2\n", "green")
        self.info_box.insert("end", "1 = No, 2 = Yes.\n\n")

        self.info_box.insert("end", "MALAISE: 1 or 2\n", "green")
        self.info_box.insert("end", "1 = No, 2 = Yes.\n\n")

        self.info_box.insert("end", "ANOREXIA: 1 or 2\n", "green")
        self.info_box.insert("end", "1 = No, 2 = Yes.\n\n")

        self.info_box.insert("end", "LIVER_BIG: 1 or 2\n", "green")
        self.info_box.insert("end", "1 = No, 2 = Yes.\n\n")

        self.info_box.insert("end", "LIVER_FIRM: 1 or 2\n", "green")
        self.info_box.insert("end", "1 = No, 2 = Yes.\n\n")

        self.info_box.insert("end", "SPLEEN_PALPABLE: 1 or 2\n", "green")
        self.info_box.insert("end", "1 = No, 2 = Yes.\n\n")

        self.info_box.insert("end", "SPIDERS: 1 or 2\n", "green")
        self.info_box.insert("end", "1 = No, 2 = Yes.\n\n")

        self.info_box.insert("end", "ASCITES: 1 or 2\n", "green")
        self.info_box.insert("end", "1 = No, 2 = Yes.\n\n")

        self.info_box.insert("end", "VARICES: 1 or 2\n", "green")
        self.info_box.insert("end", "1 = No, 2 = Yes.\n\n")

        self.info_box.insert("end", "BILIRUBIN: Range = 0.3 - 8.0\n", "green")
        self.info_box.insert("end", "Bilirubin level in blood.\n\n")

        self.info_box.insert("end", "ALK_PHOSPHATE: 26 - 295\n", "green")
        self.info_box.insert("end", "Alkaline phosphatase enzyme.\n\n")

        self.info_box.insert("end", "SGOT: 14 - 648\n", "green")
        self.info_box.insert("end", "Liver enzyme (AST).\n\n")

        self.info_box.insert("end", "ALBUMIN: Range = 2.1 - 6.4\n", "green")
        self.info_box.insert("end", "Protein level in blood.\n\n")

        self.info_box.insert("end", "PROTIME: Range = 0 - 100\n", "green")
        self.info_box.insert("end", "Prothrombin time (clotting).\n\n")

        self.info_box.insert("end", "HISTOLOGY: 1 or 2\n", "green")
        self.info_box.insert("end", "1 = No, 2 = Yes (biopsy result).\n\n")

        self.info_box.insert("end", "TARGET:\n", "green")
        self.info_box.insert("end", "Live or Die (1=Live, 2=Die)\n")
        self.info_box.configure(state="disabled")

    def _build_main_content(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=0, minsize=160)
        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=1)

        form_card = ShadowCard(main, corner_radius=20, shadow_size=4)
        form_card.grid(row=0, column=0, padx=(0, 6), pady=(0, 6), sticky="nsew")

        ctk.CTkLabel(form_card.card, text="🦠 Hepatitis Data",
                     font=ctk.CTkFont(family="Cascadia Code", size=16, weight="bold"), text_color=TEXT_MAIN).pack(
            pady=(12, 6), anchor="w", padx=15)

        form_scroll = ctk.CTkScrollableFrame(form_card.card, fg_color="transparent")
        form_scroll.pack(padx=15, pady=(0, 10), fill="both", expand=True)
        form_scroll.grid_columnconfigure(0, weight=1)
        form_scroll.grid_columnconfigure(1, weight=1)

        fields = [
            ("age", "Age", "entry"), ("sex", "Sex", "combo"),
            ("steroid", "Steroid", "combo"), ("antivirals", "Antivirals", "combo"),
            ("fatigue", "Fatigue", "combo"), ("malaise", "Malaise", "combo"),
            ("anorexia", "Anorexia", "combo"), ("liver_big", "Liver Big", "combo"),
            ("liver_firm", "Liver Firm", "combo"), ("spleen", "Spleen Palpable", "combo"),
            ("spiders", "Spiders", "combo"), ("ascites", "Ascites", "combo"),
            ("varices", "Varices", "combo"), ("bilirubin", "Bilirubin", "entry"),
            ("alk", "Alk Phosphate", "entry"), ("sgot", "SGOT", "entry"),
            ("albumin", "Albumin", "entry"), ("protime", "Protime", "entry"),
            ("histology", "Histology", "combo"),
        ]

        row = 0
        for i in range(0, len(fields), 2):
            self._create_field(form_scroll, row, 0, fields[i])
            if i + 1 < len(fields): self._create_field(form_scroll, row, 1, fields[i + 1])
            row += 1

        result_card = ShadowCard(main, corner_radius=20, shadow_size=4)
        result_card.grid(row=0, column=1, padx=(6, 0), pady=(0, 6), sticky="nsew")

        ctk.CTkLabel(result_card.card, text="🔮 Result",
                     font=ctk.CTkFont(family="Cascadia Code", size=16, weight="bold"), text_color=TEXT_MAIN).pack(
            pady=(12, 6), anchor="w", padx=15)

        inner_bg = ("#F5F3FF", "#0D0B1F")
        self.result_inner = ctk.CTkFrame(result_card.card, fg_color=inner_bg, corner_radius=14, border_width=1,
                                         border_color=PURPLE)
        self.result_inner.pack(fill="both", expand=True, padx=15, pady=(5, 10))

        ctk.CTkLabel(self.result_inner, text="Survival Prediction", font=ctk.CTkFont(family="Cascadia Code", size=12),
                     text_color=TEXT_MUTED).pack(pady=(20, 0))
        self.result_label = ctk.CTkLabel(self.result_inner, text="--",
                                         font=ctk.CTkFont(family="Cascadia Code", size=24, weight="bold"),
                                         text_color=PURPLE_DARK)
        self.result_label.pack(pady=(0, 5))
        self.unit_label = ctk.CTkLabel(self.result_inner, text="",
                                       font=ctk.CTkFont(family="Cascadia Code", size=13, weight="bold"),
                                       text_color=PURPLE)
        self.unit_label.pack(pady=(0, 10))

        self.status_label = ctk.CTkLabel(result_card.card, text="⏳ Training in background...",
                                         font=ctk.CTkFont(family="Cascadia Code", size=11), text_color=PURPLE)
        self.status_label.pack(pady=(0, 8))

        btn_frame = ctk.CTkFrame(result_card.card, fg_color="transparent")
        btn_frame.pack(fill="x", padx=15, pady=(0, 15))

        self.predict_btn = ctk.CTkButton(btn_frame, text="✨ Predict",
                                         font=ctk.CTkFont(family="Cascadia Code", size=14, weight="bold"),
                                         fg_color=PURPLE, hover_color=PURPLE_HOVER, text_color="#FFFFFF",
                                         corner_radius=12, height=40, command=self.predict, state="disabled")
        self.predict_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        ctk.CTkButton(btn_frame, text="🗑️ Clear", font=ctk.CTkFont(family="Cascadia Code", size=14),
                      fg_color="transparent", border_width=1, border_color=RED, text_color=RED,
                      hover_color=BORDER_COLOR, height=40, corner_radius=12, command=self.clear_form).pack(side="left",
                                                                                                           fill="x",
                                                                                                           expand=True)

        table_card = ShadowCard(main, corner_radius=20, shadow_size=4)
        table_card.grid(row=1, column=0, columnspan=2, padx=0, pady=(6, 0), sticky="nsew")
        ctk.CTkLabel(table_card.card, text="📈 Model Comparison",
                     font=ctk.CTkFont(family="Cascadia Code", size=14, weight="bold"), text_color=TEXT_MAIN).pack(
            pady=(8, 2), anchor="w", padx=15)

        self.tree = ttk.Treeview(table_card.card, columns=("name", "acc", "f1", "auc"), show="headings", height=3)
        self.tree.heading("name", text="Model")
        self.tree.heading("acc", text="Accuracy")
        self.tree.heading("f1", text="F1")
        self.tree.heading("auc", text="AUC")
        self.tree.column("name", width=240, anchor="center")
        self.tree.column("acc", width=90, anchor="center")
        self.tree.column("f1", width=90, anchor="center")
        self.tree.column("auc", width=90, anchor="center")
        self.tree.pack(fill="x", padx=15, pady=(0, 8))

        self.update_theme()

    def _create_field(self, parent, row, col, field_data):
        key, label, field_type = field_data
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.grid(row=row, column=col, padx=5, pady=2, sticky="ew")

        ctk.CTkLabel(f, text=label, font=ctk.CTkFont(family="Cascadia Code", size=11, weight="bold"),
                     text_color=TEXT_MAIN).pack(anchor="w", pady=(0, 3))

        if field_type == "combo":
            # استفاده از ComboBox برای مقادیر 1 و 2
            var = ctk.StringVar(value="1")
            combo = ctk.CTkComboBox(
                f, values=["1", "2"], variable=var, height=28,
                corner_radius=10, border_color=BORDER_COLOR, fg_color=BG_MAIN,
                text_color=TEXT_MAIN, font=ctk.CTkFont(family="Cascadia Code", size=12)
            )
            combo.pack(fill="x")
            self.combos[key] = var
        else:
            # استفاده از Entry برای مقادیر عددی پیوسته
            var = ctk.StringVar(value="")
            entry = ctk.CTkEntry(
                f, textvariable=var, height=28, corner_radius=10,
                border_color=BORDER_COLOR, fg_color=BG_MAIN,
                text_color=TEXT_MAIN, font=ctk.CTkFont(family="Cascadia Code", size=12)
            )
            entry.pack(fill="x")
            self.entries[key] = var

    def update_theme(self):
        mode = ctk.get_appearance_mode()
        bg_col = CARD_BG[1] if mode == "Dark" else CARD_BG[0]
        fg_col = TEXT_MAIN[1] if mode == "Dark" else TEXT_MAIN[0]
        self.info_box.configure(bg=bg_col, fg=fg_col)

        bg_col_tree = BG_MAIN[1] if mode == "Dark" else BG_MAIN[0]
        fg_col_tree = TEXT_MAIN[1] if mode == "Dark" else TEXT_MAIN[0]
        card_bg_tree = CARD_BG[1] if mode == "Dark" else CARD_BG[0]
        purple_hex = PURPLE[1] if mode == "Dark" else PURPLE[0]

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview.Heading", font=("Cascadia Code", 11, "bold"), background=bg_col_tree,
                        foreground=fg_col_tree, borderwidth=0)
        style.configure("Treeview", font=("Cascadia Code", 10), rowheight=24, background=card_bg_tree,
                        fieldbackground=card_bg_tree, foreground=fg_col_tree, borderwidth=0)
        style.map("Treeview", background=[("selected", purple_hex)])
        self.tree.tag_configure("best", background=bg_col_tree if mode == "Dark" else "#F5F3FF",
                                foreground=PURPLE_DARK[1] if mode == "Dark" else PURPLE_DARK[0])

    def show_heatmap(self):
        if self.df is not None:
            self.after(0, lambda: visuals.plot_heatmap(self.df))
        else:
            messagebox.showinfo("Heatmap", "مدل هنوز آماده نیست.")

    def show_pairplot(self):
        if self.df is not None:
            self.after(0, lambda: visuals.plot_pairplot(self.df))
        else:
            messagebox.showinfo("Plot", "مدل هنوز آماده نیست.")

    def clear_form(self):
        for var in self.entries.values(): var.set("")
        for var in self.combos.values(): var.set("1")
        self.result_label.configure(text="--")
        self.unit_label.configure(text="")

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
            self.after(200, lambda m=f"{msg}\n\n{tb[-600:] if tb else ''}": messagebox.showerror("Training Error", m))
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
        for item in self.tree.get_children(): self.tree.delete(item)
        for name, m in self.metrics.items():
            tags = ("best",) if name == self.best_name else ()
            display = f"⭐ {name}" if name == self.best_name else name
            acc = f"{m['accuracy'] * 100:.2f}%" if m.get("accuracy") is not None else "—"
            f1 = f"{m['f1']:.4f}" if m.get("f1") is not None else "—"
            auc = f"{m['auc']:.4f}" if m.get("auc") is not None else "—"
            self.tree.insert("", "end", values=(display, acc, f1, auc), tags=tags)

    def predict(self):
        if not self.model_ready:
            messagebox.showwarning("Not Ready", "مدل هنوز آماده نیست.")
            return

        try:
            vals = {k: v.get().strip() for k, v in self.entries.items()}
            if any(v == "" for v in vals.values()): raise ValueError("لطفاً همه فیلدهای متنی را پر کن.")

            # گرفتن مقادیر از ComboBox ها
            for k, v in self.combos.items():
                vals[k] = v.get().strip()

            res = pipeline.predict(
                age=float(vals["age"]), sex=float(vals["sex"]),
                steroid=float(vals["steroid"]), antivirals=float(vals["antivirals"]),
                fatigue=float(vals["fatigue"]), malaise=float(vals["malaise"]),
                anorexia=float(vals["anorexia"]), liver_big=float(vals["liver_big"]),
                liver_firm=float(vals["liver_firm"]), spleen_palpable=float(vals["spleen"]),
                spiders=float(vals["spiders"]), ascites=float(vals["ascites"]),
                varices=float(vals["varices"]), bilirubin=float(vals["bilirubin"]),
                alk_phosphate=float(vals["alk"]), sgot=float(vals["sgot"]),
                albumin=float(vals["albumin"]), protime=float(vals["protime"]),
                histology=float(vals["histology"])
            )
            self.result_label.configure(text=res["label"])
            self.unit_label.configure(text=f"Confidence: {res['confidence']:.1f}%")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"خطا در پیش‌بینی:\n{e}")