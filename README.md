# 📊 Sheydaei Data Science Final Project

A comprehensive desktop application built with Python, `CustomTkinter`, and `Scikit-Learn`. This project serves as an interactive platform to explore multiple machine learning datasets, perform Exploratory Data Analysis (EDA), and run predictive models (Regression, Classification, and Clustering) through a modern, user-friendly graphical interface.

## ✨ Features

- **Modern UI:** Built with `CustomTkinter` featuring a responsive dark/light mode toggle, sleek cards, and a clean layout.
- **Multi-Dataset Support:** Seamlessly switch between different datasets (e.g., Room Occupancy, Census Income, Liver Disorders, etc.).
- **Exploratory Data Analysis (EDA):** Built-in interactive charts (Heatmaps, Confusion Matrices, Parity Plots, and Class Distributions) using `Matplotlib`.
- **Automated Machine Learning Pipelines:** 
  - Background training with `threading` to keep the UI responsive.
  - Automatic selection of the best model based on appropriate metrics (F1-Score for Classification, RMSE/R² for Regression).
- **Real-time Predictions:** User-friendly forms with "Random Fill" capabilities to test the models instantly.
- **Model Comparison:** A dynamic table (`ttk.Treeview`) displaying the performance metrics of all trained models (Logistic Regression, Random Forest, Gradient Boosting, etc.).

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **GUI Framework:** CustomTkinter, Tkinter
- **Machine Learning:** Scikit-Learn
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib
- **Model Serialization:** Joblib

## 📂 Project Structure

```text
Sheydaei_Data_Science_Final_Project/
│
├── assets/                  # Icons, images, and UI assets
├── datasets/                # CSV data files for all models
├── forms/                   # UI forms for each dataset (e.g., room_occupancy_form.py)
├── models/                  # Saved trained models (.pkl files generated automatically)
├── pipelines/               # Machine learning training and prediction logic
├── utils/                   # Helper functions (visuals.py, etc.)
├── main.py                  # Application entry point and UI routing
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```
🚀 Installation & Setup
To run this project locally, follow these steps:

cd your-repo-name
Create a virtual environment (Recommended):

python -m venv .venv
Activate it:

Windows: .\.venv\Scripts\activate

macOS/Linux: source .venv/bin/activate

Install dependencies:


pip install -r requirements.txt
Run the application:

python main.py
📦 Building the Executable (Windows)
To create a standalone .exe file that can be distributed without requiring Python to be installed, use PyInstaller. We use a .spec file to bundle the datasets and assets folders correctly.

pyinstaller app.spec
The executable will be available in the dist/ directory
