import os
import customtkinter as ctk
from PIL import Image, ImageDraw

# ایمپورت فرم‌ دیتاست‌ها
from forms.abalone_form import AbaloneFormPage
from forms.user_knowledge_form import UserKnowledgeFormPage
from forms.real_estate_form import RealEstateFormPage
from forms.wifi_form import WifiFormPage
from forms.car_form import CarFormPage
from forms.fertility_form import FertilityFormPage
from forms.bankruptcy_form import BankruptcyFormPage
from forms.auto_mpg_form import AutoMpgFormPage
from forms.heart_disease_form import HeartDiseaseFormPage
from forms.daily_demand_form import DailyDemandFormPage
from forms.blood_donation_form import BloodDonationFormPage
from forms.pollution_form import PollutionFormPage
from forms.echocardiogram_form import EchocardiogramFormPage
from forms.concrete_form import ConcreteFormPage
from forms.liver_form import LiverFormPage
from forms.dow_jones_form import DowJonesFormPage
from forms.energy_form import EnergyFormPage
from forms.glass_form import GlassFormPage
from forms.hepatitis_form import HepatitisFormPage
from forms.wholesale_form import WholesaleFormPage
from forms.travel_reviews_form import TravelReviewsFormPage
from forms.istanbul_form import IstanbulFormPage
from forms.bike_sharing_form import BikeSharingFormPage
from forms.room_occupancy_form import RoomOccupancyFormPage
from forms.census_income_form import CensusIncomeFormPage
from forms.covid19_form import Covid19FormPage
from forms.asd_form import ASDFormPage
from forms.credit_default_form import CreditDefaultFormPage
from forms.banknote_form import BanknoteFormPage

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# تنظیمات تم و رنگ‌ها (سفید / بنفش)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
ctk.set_appearance_mode("dark")   #  دارک مود پیش‌فرض
ctk.set_default_color_theme("blue")

BG_MAIN       = ("#F5F3FF", "#0A0A10")   # پس‌زمینه اصلی
CARD_BG       = ("#FFFFFF", "#151520")   # کارت‌ها
CARD_SHADOW   = ("#E9E5FF", "#000000")   # سایه
BORDER_COLOR  = ("#EDE9FE", "#2A2A35")
PURPLE        = ("#7C3AED", "#8B5CF6")
PURPLE_DARK   = ("#5B21B6", "#6D28D9")
PURPLE_HOVER  = ("#6D28D9", "#7C3AED")
TEXT_MAIN     = ("#1F2937", "#FFFFFF")
TEXT_MUTED    = ("#6B7280", "#9CA3AF")
GREEN         = ("#059669", "#10B981")
RED           = ("#DC2626", "#EF4444")

FONT_TECH_HEADER = ("Cascadia Code", 26, "bold")
FONT_TECH_TITLE  = ("Cascadia Code", 18, "bold")
FONT_TECH_BODY   = ("Cascadia Code", 14)


def bind_click_recursively(widget, command):
    widget.bind("<Button-1>", lambda e: command())
    for child in widget.winfo_children():
        bind_click_recursively(child, command)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# کلاس اصلی برنامه
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class DSApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Machine Learning Project _ Dina Sheydaei")
        self.geometry("1100x750")
        self.resizable(False, False)
        self.configure(fg_color=BG_MAIN)

        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (1100 // 2)
        y = (self.winfo_screenheight() // 2) - (750 // 2)
        self.geometry(f"1100x750+{x}+{y}")

        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.ico")
        if os.path.exists(icon_path):
            try:
                self.iconbitmap(icon_path)
            except:
                pass

        self.selected_task = None
        self.selected_dataset = None
        self.history = []

        self._build_header()

        self.container = ctk.CTkFrame(self, fg_color="transparent")
        self.container.pack(side="top", fill="both", expand=True, padx=10, pady=10)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}


        # اضافه کردن تمام کلاس‌ها به برنامه
        for F in (
            WelcomePage, TaskSelectionPage, DatasetSelectionPage, PlaceholderPage,
            AbaloneFormPage, UserKnowledgeFormPage, RealEstateFormPage,
            WifiFormPage, CarFormPage, FertilityFormPage, BankruptcyFormPage,
            AutoMpgFormPage, HeartDiseaseFormPage, DailyDemandFormPage,
            BloodDonationFormPage, PollutionFormPage,
            EchocardiogramFormPage, ConcreteFormPage, LiverFormPage, DowJonesFormPage, EnergyFormPage, GlassFormPage,
            HepatitisFormPage, WholesaleFormPage, TravelReviewsFormPage,IstanbulFormPage,
            BikeSharingFormPage,RoomOccupancyFormPage,CensusIncomeFormPage,Covid19FormPage,
            ASDFormPage,CreditDefaultFormPage,BanknoteFormPage,
        ):
            name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.frames[name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("WelcomePage", add_to_history=False)

    def _build_header(self):
        self.header = ctk.CTkFrame(self, fg_color=CARD_BG, height=60, corner_radius=0, border_width=1,
                                   border_color=BORDER_COLOR)
        self.header.pack(side="top", fill="x")
        self.header.pack_propagate(False)

        # logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.png")
        # if os.path.exists(logo_path):
        #     try:
        #         pil_logo = Image.open(logo_path)
        #         ctk_logo = ctk.CTkImage(light_image=pil_logo, dark_image=pil_logo, size=(35, 35))
        #         ctk.CTkLabel(self.header, text="", image=ctk_logo).pack(side="left", padx=(20, 10), pady=10)
        #     except:
        #         pass

        ctk.CTkLabel(self.header, text="   Machine Learning Project",
                     font=ctk.CTkFont(family="Cascadia Code", size=20, weight="bold"),
                     text_color=PURPLE).pack(side="left", pady=10)

        right_controls = ctk.CTkFrame(self.header, fg_color="transparent")
        right_controls.pack(side="right", padx=20)

        self.back_btn = ctk.CTkButton(right_controls, text="BACK",
                                      font=ctk.CTkFont(family="Cascadia Code", size=14, weight="bold"),
                                      fg_color=PURPLE, hover_color=PURPLE_HOVER,
                                      text_color="white", corner_radius=8,
                                      width=80, height=30, command=self.go_back)
        self.back_btn.pack(side="left", padx=(0, 15))

        self.theme_switch = ctk.CTkSwitch(right_controls, text="Dark Mode", command=self.toggle_theme,
                                          font=ctk.CTkFont(family="Cascadia Code", size=12),
                                          progress_color=PURPLE, button_color=PURPLE, text_color=TEXT_MAIN)
        self.theme_switch.select()
        self.theme_switch.pack(side="left")

    def go_back(self):
        if len(self.history) > 1:
            self.history.pop()
            previous_page = self.history[-1]
            self.show_frame(previous_page, add_to_history=False)
        else:
            self.show_frame("WelcomePage", add_to_history=False)

    def toggle_theme(self):
        current_mode = ctk.get_appearance_mode()
        if current_mode == "Dark":
            ctk.set_appearance_mode("light")
            self.theme_switch.configure(text="Light Mode")
        else:
            ctk.set_appearance_mode("dark")
            self.theme_switch.configure(text="Dark Mode")


        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # اضافه کردن فرم‌های جدید به این لیست
        for page in [
            "AbaloneFormPage", "UserKnowledgeFormPage", "RealEstateFormPage", "WifiFormPage",
            "CarFormPage", "FertilityFormPage", "BankruptcyFormPage", "AutoMpgFormPage",
            "HeartDiseaseFormPage", "DailyDemandFormPage", "BloodDonationFormPage",
            "PollutionFormPage", "EchocardiogramFormPage", "ConcreteFormPage", "DowJonesFormPage", "EnergyFormPage",
            "GlassFormPage" ,"HepatitisFormPage", "WholesaleFormPage" , "TravelReviewsFormPage"
            , "IstanbulFormPage","BikeSharingFormPage","RoomOccupancyFormPage","CensusIncomeFormPage",
            "Covid19FormPage","ASDFormPage","CreditDefaultFormPage","BanknoteFormPage",

        ]:
            if page in self.frames:
                try:
                    self.frames[page].update_theme()
                except:
                    pass

    def show_frame(self, name, add_to_history=True):
        if add_to_history:
            if not self.history or self.history[-1] != name:
                self.history.append(name)
        self.frames[name].tkraise()

    def set_task(self, task):
        self.selected_task = task

    def set_dataset(self, dataset):
        self.selected_dataset = dataset


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# (Welcome)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class WelcomePage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # کارت اصلی
        main_card = ctk.CTkFrame(self, fg_color=CARD_BG, corner_radius=25, border_width=0,
                                 width=900, height=500)
        main_card.grid(row=0, column=0)
        main_card.grid_propagate(False)

        # ستون عکس  (۴۸۰) و ستون فرم (۴۲۰)
        main_card.grid_columnconfigure(0, weight=0, minsize=480)
        main_card.grid_columnconfigure(1, weight=1)
        main_card.grid_rowconfigure(0, weight=1)

        # سمت چپ: عکس
        left_frame = ctk.CTkFrame(main_card, fg_color=CARD_BG, corner_radius=25)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 0), pady=(0, 0))

        assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
        img_path = os.path.join(assets_dir, "welcome.png")

        if os.path.exists(img_path):
            try:
                target_width, target_height = 480, 500

                pil_img = Image.open(img_path).convert("RGBA")


                ratio = max(target_width / pil_img.width, target_height / pil_img.height)
                new_size = (int(pil_img.width * ratio), int(pil_img.height * ratio))
                pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)

                left = (pil_img.width - target_width) / 2
                top = (pil_img.height - target_height) / 2
                right = (pil_img.width + target_width) / 2
                bottom = (pil_img.height + target_height) / 2
                pil_img = pil_img.crop((left, top, right, bottom))

                mask = Image.new("L", (target_width, target_height), 0)
                draw = ImageDraw.Draw(mask)

                draw.rounded_rectangle((0, 0, target_width, target_height), radius=25, fill=255)
                draw.rectangle((25, 0, target_width, target_height), fill=255)

                rounded_img = pil_img.copy()
                rounded_img.putalpha(mask)

                ctk_img = ctk.CTkImage(light_image=rounded_img, dark_image=rounded_img,
                                       size=(target_width, target_height))

                #  عکس به اندازه‌ی کل فریم
                img_label = ctk.CTkLabel(left_frame, text="", image=ctk_img)
                img_label.place(relx=0, rely=0, relwidth=1, relheight=1)

            except Exception:
                ctk.CTkLabel(left_frame, text="🪐", font=("Segoe UI Emoji", 100), text_color=PURPLE).pack(expand=True)
        else:
            ctk.CTkLabel(left_frame, text="🪐", font=("Segoe UI Emoji", 100), text_color=PURPLE).pack(expand=True)

        # سمت راست: فرم لاگین
        right_frame = ctk.CTkFrame(main_card, fg_color="transparent")
        right_frame.grid(row=0, column=1, sticky="nsew", padx=40, pady=30)

        ctk.CTkLabel(right_frame, text="Welcome to the Data World",
                     font=ctk.CTkFont(family="Cascadia Code", size=24, weight="bold"),
                     text_color=TEXT_MAIN).pack(pady=(40, 5))

        ctk.CTkLabel(right_frame, text="Please log in to continue",
                     font=ctk.CTkFont(family="Cascadia Code", size=12),
                     text_color=TEXT_MUTED).pack(pady=(0, 30))

        user_bg = ("#F3E8FF", "#1E1B2E")

        # فیلد نام کاربری
        ctk.CTkLabel(right_frame, text="Username", font=ctk.CTkFont(family="Cascadia Code", size=11, weight="bold"),
                     text_color=TEXT_MAIN).pack(anchor="w", pady=(0, 5))
        self.username_entry = ctk.CTkEntry(right_frame, height=45, corner_radius=10, fg_color=user_bg,
                                           text_color=TEXT_MAIN, border_width=0,
                                           placeholder_text="👤  Enter username",
                                           placeholder_text_color=TEXT_MUTED,
                                           font=ctk.CTkFont(family="Cascadia Code", size=13))
        self.username_entry.pack(fill="x", pady=(0, 20))

        # فیلد رمز عبور
        ctk.CTkLabel(right_frame, text="Password", font=ctk.CTkFont(family="Cascadia Code", size=11, weight="bold"),
                     text_color=TEXT_MAIN).pack(anchor="w", pady=(0, 5))
        self.password_entry = ctk.CTkEntry(right_frame, height=45, corner_radius=10, fg_color=user_bg,
                                           text_color=TEXT_MAIN, border_width=0, show="*",
                                           placeholder_text="🔒  Enter password",
                                           placeholder_text_color=TEXT_MUTED,
                                           font=ctk.CTkFont(family="Cascadia Code", size=13))
        self.password_entry.pack(fill="x", pady=(0, 10))

        self.error_label = ctk.CTkLabel(right_frame, text="",
                                        font=ctk.CTkFont(family="Cascadia Code", size=12, weight="bold"),
                                        text_color=RED)
        self.error_label.pack(pady=(0, 15))

        # دکمه لاگین
        ctk.CTkButton(right_frame, text="Login",
                      font=ctk.CTkFont(family="Cascadia Code", size=14, weight="bold"),
                      fg_color=PURPLE, hover_color=PURPLE_HOVER, corner_radius=10,
                      height=45, text_color="#FFFFFF",
                      command=self.login).pack(fill="x", pady=(0, 20))

        ctk.CTkLabel(right_frame, text="Developed By Dina Sheydaei",
                     font=ctk.CTkFont(family="Cascadia Code", size=10),
                     text_color=GREEN).pack(side="bottom", pady=(0, 0))

    def login(self):
        self.error_label.configure(text="")

        # دریافت مقادیر از فیلدها
        username = self.username_entry.get().strip()
        password = self.password_entry.get().strip()

        # چک کردن خالی نبودن فیلدها
        if not username or not password:
            self.error_label.configure(text="Please enter both username and password.", text_color=RED)
            return

        # بررسی نام کاربری و رمز عبور
        if username != "cosmicml" or password != "1234":
            self.error_label.configure(text="Invalid username or password.", text_color=RED)
            return

        # در صورت صحیح بودن، وارد برنامه می‌شود
        self.controller.show_frame("TaskSelectionPage")


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Task Selection
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class TaskSelectionPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        center_frame = ctk.CTkFrame(self, fg_color=CARD_BG, corner_radius=20, border_width=1, border_color=BORDER_COLOR)
        center_frame.grid(row=0, column=0, ipadx=30, ipady=30)

        # عنوان و زیرعنوان
        ctk.CTkLabel(center_frame, text="Meet Your Model", font=FONT_TECH_HEADER, text_color=TEXT_MAIN).pack(
            pady=(10, 5))
        ctk.CTkLabel(center_frame, text="Explore algorithms. Discover possibilities.",
                     font=ctk.CTkFont(family="Cascadia Code", size=14), text_color=TEXT_MUTED).pack(
            pady=(0, 30))

        btn_frame = ctk.CTkFrame(center_frame, fg_color="transparent")
        btn_frame.pack(pady=10)

        tasks = [("Clustering", "Group similar data", "clustering"),
                 ("Classification", "Predict labels", "classification"),
                 ("Regression", "Predict values", "regression")]

        assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
        for title, desc, key in tasks:
            card = ctk.CTkFrame(btn_frame, fg_color=BG_MAIN, corner_radius=20, border_width=1,
                                border_color=BORDER_COLOR, width=240, height=350)
            card.pack(side="left", padx=20)
            card.pack_propagate(False)
            img_path = None
            for ext in [".png", ".jpg", ".jpeg"]:
                temp_path = os.path.join(assets_dir, key + ext)
                if os.path.exists(temp_path):
                    img_path = temp_path
                    break
            if img_path:
                try:
                    pil_img = Image.open(img_path)
                    target_size = 180
                    pil_img = pil_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
                    mask = Image.new("L", (target_size, target_size), 0)
                    draw = ImageDraw.Draw(mask)
                    draw.rounded_rectangle((0, 0, target_size, target_size), radius=20, fill=255)
                    rounded_img = pil_img.convert("RGBA")
                    rounded_img.putalpha(mask)
                    ctk_img = ctk.CTkImage(light_image=rounded_img, dark_image=rounded_img,
                                           size=(target_size, target_size))
                    ctk.CTkLabel(card, text="", image=ctk_img).pack(pady=(20, 15))
                except:
                    ctk.CTkLabel(card, text="📊", font=("Segoe UI Emoji", 50)).pack(pady=(40, 20))
            else:
                ctk.CTkLabel(card, text="📊", font=("Segoe UI Emoji", 50)).pack(pady=(40, 20))

            ctk.CTkLabel(card, text=title.upper(), font=ctk.CTkFont(family="Cascadia Code", size=18, weight="bold"),
                         text_color=TEXT_MAIN).pack(pady=(5, 5))
            ctk.CTkLabel(card, text=desc, font=ctk.CTkFont(family="Cascadia Code", size=13), text_color=TEXT_MUTED,
                         wraplength=200).pack(pady=(0, 20))
            bind_click_recursively(card, lambda t=key: self.go(t))

    def go(self, task):
        self.controller.set_task(task)
        self.controller.show_frame("DatasetSelectionPage")
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# (Dataset Selection)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class DatasetSelectionPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.center_frame = ctk.CTkFrame(self, fg_color=CARD_BG, corner_radius=20, border_width=1,
                                         border_color=BORDER_COLOR)
        self.center_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.scroll_frame = ctk.CTkScrollableFrame(self.center_frame, fg_color="transparent")
        self.scroll_frame.pack(fill="both", expand=True, padx=10, pady=(10, 0))

    def tkraise(self, *args, **kwargs):
        super().tkraise(*args, **kwargs)
        self.render()

    def render(self):
        for widget in self.center_frame.winfo_children(): widget.destroy()
        self.scroll_frame = ctk.CTkScrollableFrame(self.center_frame, fg_color="transparent")
        self.scroll_frame.pack(fill="both", expand=True, padx=10, pady=(10, 0))
        task = self.controller.selected_task
        if not task: return
        datasets_map = {
            "regression": [
                {"en": "Abalone Age", "img": "abalone", "page": "AbaloneFormPage"},
                {"en": "House Price", "img": "house", "page": "RealEstateFormPage"},
                {"en": "Fuel Efficiency", "img": "fuel", "page": "AutoMpgFormPage"},
                {"en": "Order Demand", "img": "demand", "page": "DailyDemandFormPage"},
                {"en": "Pollution Level", "img": "pollution", "page": "PollutionFormPage"},
                {"en": "Concrete Strength", "img": "concrete", "page": "ConcreteFormPage"},
                {"en": "Stock Return", "img": "stock", "page": "DowJonesFormPage"},
                {"en": "Building Load", "img": "building", "page": "EnergyFormPage"},
                {"en": "Istanbul Stock", "img": "istanbul", "page": "IstanbulFormPage"},
                {"en": "Bike Rental", "img": "bike", "page": "BikeSharingFormPage"},

            ],
            "classification": [
                {"en": "Student Knowledge", "img": "student", "page": "UserKnowledgeFormPage"},
                {"en": "WiFi Localization", "img": "wifi", "page": "WifiFormPage"},
                {"en": "Car Evaluation", "img": "car", "page": "CarFormPage"},
                {"en": "Fertility Quality", "img": "fertility", "page": "FertilityFormPage"},
                {"en": "Bankruptcy Risk", "img": "bankruptcy", "page": "BankruptcyFormPage"},
                {"en": "Heart Disease", "img": "heart", "page": "HeartDiseaseFormPage"},
                {"en": "Blood Donation", "img": "blood", "page": "BloodDonationFormPage"},
                {"en": "Heart Attack Surv.", "img": "survival", "page": "EchocardiogramFormPage"},
                {"en": "Glass Type", "img": "glass", "page": "GlassFormPage"},
                {"en": "Hepatitis Survival", "img": "hepatitis", "page": "HepatitisFormPage"},
                {"en": "Room Occupancy", "img": "room", "page": "RoomOccupancyFormPage"},
                {"en": "Census Income", "img": "income", "page": "CensusIncomeFormPage"},
                {"en": "COVID-19", "img": "covid", "page": "Covid19FormPage"},
                {"en": "ASD Cases", "img": "asd", "page": "ASDFormPage"},
                {"en": "Default Probability", "img": "default", "page": "CreditDefaultFormPage"},
                {"en": "Banknote Genuine", "img": "banknote", "page": "BanknoteFormPage"},
            ],
            "clustering": [
                {"en": "Wholesale Spending", "img": "wholesale", "page": "WholesaleFormPage"},
                {"en": "Liver Disorder", "img": "liver", "page": "LiverFormPage"},
                {"en": "Travel Reviews", "img": "travel", "page": "TravelReviewsFormPage"},
            ]
        }

        # بخش سوم
        ctk.CTkLabel(self.scroll_frame, text=f"{task.upper()} LAB",
                     font=ctk.CTkFont(family="Cascadia Code", size=32, weight="bold"), text_color=PURPLE).pack(
            pady=(10, 30))
        grid_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        grid_frame.pack(fill="both", expand=True)
        for i in range(3): grid_frame.grid_columnconfigure(i, weight=1)
        assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "datasets")
        for idx, ds in enumerate(datasets_map.get(task, [])):
            r = idx // 3
            col = idx % 3
            card = ctk.CTkFrame(grid_frame, fg_color=BG_MAIN, corner_radius=20, border_width=1,
                                border_color=BORDER_COLOR, width=300, height=280)
            card.grid(row=r, column=col, padx=20, pady=20)
            card.grid_propagate(False)
            img_path = None
            for ext in [".png", ".jpg", ".jpeg"]:
                temp_path = os.path.join(assets_dir, ds["img"] + ext)
                if os.path.exists(temp_path):
                    img_path = temp_path
                    break
            target_width, target_height = 240, 180
            if img_path:
                try:
                    pil_img = Image.open(img_path).convert("RGBA")
                    ratio = max(target_width / pil_img.width, target_height / pil_img.height)
                    new_size = (int(pil_img.width * ratio), int(pil_img.height * ratio))
                    pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
                    left = (pil_img.width - target_width) / 2
                    top = (pil_img.height - target_height) / 2
                    right = (pil_img.width + target_width) / 2
                    bottom = (pil_img.height + target_height) / 2
                    pil_img = pil_img.crop((left, top, right, bottom))
                    mask = Image.new("L", (target_width, target_height), 0)
                    draw = ImageDraw.Draw(mask)
                    draw.rounded_rectangle((0, 0, target_width, target_height), radius=15, fill=255)
                    rounded_img = pil_img.copy()
                    rounded_img.putalpha(mask)
                    img_frame = ctk.CTkFrame(card, fg_color=BORDER_COLOR, corner_radius=15, width=target_width,
                                             height=target_height)
                    img_frame.pack(pady=(20, 15), padx=20)
                    img_frame.pack_propagate(False)
                    ctk_img = ctk.CTkImage(light_image=rounded_img, dark_image=rounded_img,
                                           size=(target_width, target_height))
                    img_label = ctk.CTkLabel(img_frame, text="", image=ctk_img, fg_color="transparent")
                    img_label.pack(expand=True, fill="both", padx=3, pady=3)
                except:
                    ctk.CTkLabel(card, text="🖼️", font=("Segoe UI Emoji", 50)).pack(pady=(40, 20))
            else:
                ctk.CTkLabel(card, text="🖼️", font=("Segoe UI Emoji", 50)).pack(pady=(40, 20))
            ctk.CTkLabel(card, text=ds['en'].upper(), font=ctk.CTkFont(family="Cascadia Code", size=16, weight="bold"),
                         text_color=PURPLE, wraplength=270).pack(pady=(5, 10))
            bind_click_recursively(card, lambda d=ds: self.open_form(d))

    def open_form(self, ds):
        self.controller.set_dataset(ds["en"])
        self.controller.show_frame(ds["page"])


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Placeholder
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class PlaceholderPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        center_frame = ctk.CTkFrame(self, fg_color=CARD_BG, corner_radius=20, border_width=1, border_color=BORDER_COLOR)
        center_frame.grid(row=0, column=0, ipadx=50, ipady=50)
        self.label = ctk.CTkLabel(center_frame, text="", font=FONT_TECH_HEADER, text_color=TEXT_MUTED, justify="center")
        self.label.pack(pady=20)

    def tkraise(self, *args, **kwargs):
        super().tkraise(*args, **kwargs)
        ds = self.controller.selected_dataset
        self.label.configure(text=f"DATASET: {ds}\n\nMODULE UNDER CONSTRUCTION...")


if __name__ == "__main__":
    app = DSApp()
    app.mainloop()