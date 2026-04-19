# 📉 Customer Churn Prediction & Retention System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)](https://scikit-learn.org/)
[![Status: In-Development](https://img.shields.io/badge/Status-In--Development-green.svg)]()

This repository contains a data-driven system designed to identify customers at risk of leaving (churning) and provide actionable insights for retention. Currently, the project focuses on high-accuracy predictive modeling, with an **Agentic AI** layer planned for automated outreach.

## Project Goals
1. **Analyze:** Perform deep Exploratory Data Analysis (EDA) to find churn drivers.
2. **Predict:** Build a robust Machine Learning pipeline to classify churn risk.
3. **Automate (Future):** Implement an Agentic workflow to draft personalized retention offers.

## Tech Stack
- **Data Handling:** `Pandas`, `NumPy`
- **Visualization:** `Matplotlib`, `Seaborn`
- **Machine Learning:** `Scikit-Learn`, `RandomForest`, `K-means`
- **Environment:** `Jupyter Notebook` / `Python 3.x`

## How to Run the Pipeline

1. **Preprocess the Data**: Open the `notebooks/preprocessing.ipynb` in VS Code or Jupyter, select your `.venv` kernel, and run all cells. This will clean the raw data and export it to `data/processed_churn_data.parquet`.

2. **Train the Model**: Run the dedicated training script to train the model, evaluate it via cross-validation, and save the artifact.
```bash
python src/train.py
```

3. **Run the Streamlit App**: Start the interactive web application.
```bash
streamlit run app.py
```

## Model Performance

Based on the 5-fold cross-validation and test set evaluation using the `RandomForestClassifier`:

- **Mean CV Accuracy**: 78.76% (± 0.22%)
- **Test Set Accuracy**: 77.04%
- **ROC-AUC Score**: 82.73%

## Project Structure
```text
├── data/               # Raw and processed datasets
├── models/             # Saved ML model artifacts (*.pkl)
├── notebooks/          # EDA and Model Training experiments
├── src/                # Modular Python scripts
│   ├── preprocessing.py # Feature engineering & cleaning
│   ├── train.py         # Dedicated training script w/ cross-validation
│   ├── ui.py            # Streamlit UI layouts
│   ├── callbacks.py     # Streamlit app logic wrapper
│   └── inference.py     # Model inference orchestration
├── app.py              # Main Streamlit Entrypoint
├── requirements.txt    # Project dependencies
└── README.md
```
---

##  Setup & Installation

Follow these steps to replicate the environment on your local machine:

### 1. Clone the Repository
```bash
git clone https://github.com/Vegapunk-debug/customer-churn-agentic-retention-system.git
cd customer-churn-agentic-retention-system
```
### 2. Create and Activate Virtual Environment
```bash
# For macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```


