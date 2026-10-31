import os
import traceback
import threading
import random
import tkinter as tk
from tkinter import messagebox, ttk
import customtkinter as ctk
import joblib

from pipelines import wholesale_clustering as pipeline
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


class WholesaleFormPage(ctk.CTkFrame):
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

        self.info_box.insert("end", "Find patterns from spending data at wholesale.\n")
        self.info_box.insert("end", "────────────────────────\n\n")
        self.info_box.insert("end", "INPUT FIELDS & RANGES:\n\n", "green")

        self.info_box.insert("end", "Channel: 1 or 2\n", "green")
        self.info_box.insert("end", "1 = Horeca, 2 = Retail.\n\n")

        self.info_box.insert("end", "Region: 1, 2, or 3\n", "green")
        self.info_box.insert("end", "Geographic region of client.\n\n")

        self.info_box.insert("end", "Fresh: 3 - 112151\n", "green")
        self.info_box.insert("end", "Annual spending on Fresh products.\n\n")

        self.info_box.insert("end", "Milk: 55 - 73498\n", "green")
        self.info_box.insert("end", "Annual spending on Milk.\n\n")

        self.info_box.insert("end", "Grocery: 3 - 92780\n", "green")
        self.info_box.insert("end", "Annual spending on Grocery.\n\n")

        self.info_box.insert("end", "Frozen: 25 - 60869\n", "green")
        self.info_box.insert("end", "Annual spending on Frozen goods.\n\n")

        self.info_box.insert("end", "Detergents_Paper: 3 - 40827\n", "green")
        self.info_box.insert("end", "Annual spending on Detergents.\n\n")

        self.info_box.insert("end", "Delicassen: 3 - 47943\n", "green")
        self.info_box.insert("end", "Annual spending on Delicatessen.\n\n")

        self.info_box.insert("end", "TARGET:\n", "green")
        self.info_box.insert("end", "Customer Cluster Group\n")
        self.info_box.configure(state="disabled")

    def _build_main_content(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=0, minsize=160)
        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=1)

        # --- ستون چپ: فرم ---
        form_card = ShadowCard(main, corner_radius=20, shadow_size=4)
        form_card.grid(row=0, column=0, padx=(0, 6), pady=(0, 6), sticky="nsew")

        ctk.CTkLabel(form_card.card, text="🛒 Wholesale Spending",
                     font=ctk.CTkFont(family="Cascadia Code", size=16, weight="bold"), text_color=TEXT_MAIN).pack(
            pady=(12, 6), anchor="w", padx=15)

        form_scroll = ctk.CTkScrollableFrame(form_card.card, fg_color="transparent")
        form_scroll.pack(padx=15, pady=(0, 5), fill="both", expand=True)
        form_scroll.grid_columnconfigure(0, weight=1)
        form_scroll.grid_columnconfigure(1, weight=1)

        fields = [
            ("fresh", "Fresh"), ("milk", "Milk"),
            ("grocery", "Grocery"), ("frozen", "Frozen"),
            ("detergents", "Detergents Paper"), ("delicassen", "Delicassen"),
        ]

        row = 0
        for i in range(0, len(fields), 2):
            self._create_field(form_scroll, row, 0, fields[i])
            if i + 1 < len(fields): self._create_field(form_scroll, row, 1, fields[i + 1])
            row += 1

        row += 1
        self._create_combo(form_scroll, row, 0, "channel", "Channel", ["1", "2"])
        self._create_combo(form_scroll, row, 1, "region", "Region", ["1", "2", "3"])

        # --- دکمه‌های زیر فرم ---
        btn_frame_form = ctk.CTkFrame(form_card.card, fg_color="transparent")
        btn_frame_form.pack(fill="x", padx=15, pady=(0, 15))

        # دکمه Random Fill (سبز)
        ctk.CTkButton(
            btn_frame_form,
            text="🎲 Random Fill",
            font=ctk.CTkFont(family="Cascadia Code", size=12, weight="bold"),
            fg_color="transparent",
            border_width=1,
            border_color=GREEN,
            text_color=GREEN,
            hover_color=BORDER_COLOR,
            corner_radius=10,
            height=32,
            command=self.fill_random
        ).pack(side="left", fill="x", expand=True, padx=(0, 5))

        # دکمه Clear (قرمز)
        ctk.CTkButton(
            btn_frame_form,
            text="🗑️ Clear",
            font=ctk.CTkFont(family="Cascadia Code", size=12),
            fg_color="transparent",
            border_width=1,
            border_color=RED,
            text_color=RED,
            hover_color=BORDER_COLOR,
            corner_radius=10,
            height=32,
            command=self.clear_form
        ).pack(side="left", fill="x", expand=True)

        # --- ستون راست: نتیجه ---
        result_card = ShadowCard(main, corner_radius=20, shadow_size=4)
        result_card.grid(row=0, column=1, padx=(6, 0), pady=(0, 6), sticky="nsew")

        ctk.CTkLabel(result_card.card, text="🔮 Result",
                     font=ctk.CTkFont(family="Cascadia Code", size=16, weight="bold"), text_color=TEXT_MAIN).pack(
            pady=(12, 6), anchor="w", padx=15)

        inner_bg = ("#F5F3FF", "#0D0B1F")
        self.result_inner = ctk.CTkFrame(result_card.card, fg_color=inner_bg, corner_radius=14, border_width=1,
                                         border_color=PURPLE)
        self.result_inner.pack(fill="both", expand=True, padx=15, pady=(5, 10))

        ctk.CTkLabel(self.result_inner, text="Customer Segment", font=ctk.CTkFont(family="Cascadia Code", size=12),
                     text_color=TEXT_MUTED).pack(pady=(20, 0))

        # لیبل نتیجه با wraplength
        self.result_label = ctk.CTkLabel(
            self.result_inner, text="--",
            font=ctk.CTkFont(family="Cascadia Code", size=16, weight="bold"),
            text_color=PURPLE_DARK,
            wraplength=240
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

        # --- جدول مدل‌ها ---
        table_card = ShadowCard(main, corner_radius=20, shadow_size=4)
        table_card.grid(row=1, column=0, columnspan=2, padx=0, pady=(6, 0), sticky="nsew")
        ctk.CTkLabel(table_card.card, text="📈 Model Comparison",
                     font=ctk.CTkFont(family="Cascadia Code", size=14, weight="bold"), text_color=TEXT_MAIN).pack(
            pady=(8, 2), anchor="w", padx=15)

        self.tree = ttk.Treeview(table_card.card, columns=("name", "silhouette", "clusters"), show="headings", height=3)
        self.tree.heading("name", text="Model")
        self.tree.heading("silhouette", text="Silhouette Score")
        self.tree.heading("clusters", text="Clusters")
        self.tree.column("name", width=240, anchor="center")
        self.tree.column("silhouette", width=120, anchor="center")
        self.tree.column("clusters", width=90, anchor="center")
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
        self.entries["channel"].set("1")
        self.entries["region"].set("1")
        self.result_label.configure(text="--")
        self.unit_label.configure(text="")

    def fill_random(self):
        self.entries["fresh"].set(str(random.randint(3, 110000)))
        self.entries["milk"].set(str(random.randint(55, 73000)))
        self.entries["grocery"].set(str(random.randint(3, 92000)))
        self.entries["frozen"].set(str(random.randint(25, 60000)))
        self.entries["detergents"].set(str(random.randint(3, 40000)))
        self.entries["delicassen"].set(str(random.randint(3, 47000)))
        self.entries["channel"].set(random.choice(["1", "2"]))
        self.entries["region"].set(random.choice(["1", "2", "3"]))

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
            sil = f"{m['silhouette']:.4f}" if m.get("silhouette") is not None else "—"
            clus = f"{m['n_clusters']}" if m.get("n_clusters") is not None else "—"
            self.tree.insert("", "end", values=(display, sil, clus), tags=tags)

    def predict(self):
        if not self.model_ready:
            messagebox.showwarning("Not Ready", "مدل هنوز آماده نیست.")
            return

        try:
            vals = {k: v.get().strip() for k, v in self.entries.items()}
            if any(v == "" for v in vals.values()): raise ValueError("لطفاً همه فیلدها را پر کن.")

            res = pipeline.predict(
                channel=int(float(vals["channel"])), region=int(float(vals["region"])),
                fresh=float(vals["fresh"]), milk=float(vals["milk"]),
                grocery=float(vals["grocery"]), frozen=float(vals["frozen"]),
                detergents=float(vals["detergents"]), delicassen=float(vals["delicassen"])
            )
            self.result_label.configure(text=res["label"])
            self.unit_label.configure(text="")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"خطا در پیش‌بینی:\n{e}")