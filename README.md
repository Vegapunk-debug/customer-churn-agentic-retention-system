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

## Project Structure
```text
├── data/               # Raw and processed datasets
├── notebooks/          # EDA and Model Training experiments
│   └── churn_analysis.ipynb
├── src/                # Modular Python scripts
│   ├── preprocessing.py # Feature engineering & cleaning
│   └── model.py         # Model training and evaluation
├── requirements.txt    # Project dependencies
└── README.md
```
---

##  Setup & Installation

Follow these steps to replicate the environment on your local machine:

### 1. Clone the Repository
```bash
git clone [https://github.com/Vegapunk-debug/customer-churn-agentic-retention-system.git](https://github.com/Vegapunk-debug/customer-churn-agentic-retention-system.git)
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

## How to Run the Pipeline

1. Open the Preprocessing Notebook: Navigate to ```bash notebooks/preprocessing.ipynb ``` in VS Code.

2. Select Kernel: Ensure the kernel is set to your .venv

3. Run All Cells.

4. The notebook will automatically: Download the raw data into data/raw_churn_data.csv.

5. Clean and encode the features.

6. It will automatically export the result to ``` data/processed_churn_data.parquet ```


