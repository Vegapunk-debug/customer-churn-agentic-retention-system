# Customer Churn Prediction and Agentic Retention Strategy System

## Project Overview
This repository contains an end-to-end artificial intelligence system designed to identify at-risk customers and generate structured, data-driven retention strategies. The project is divided into two primary phases: a machine learning pipeline for churn classification and an agentic AI assistant for personalized intervention planning.

## Milestone 1: Data Preprocessing & Pipeline
The current phase focuses on building a robust automated data pipeline that transforms raw customer data into a machine-learning-ready format.

### Key Features
* **Automated Ingestion:** Uses `kagglehub` to fetch the latest Telco Churn dataset directly [cite: 5, 2026-02-01].
* **Data Cleaning:** Handles missing values in `TotalCharges` and removes non-predictive features like `customerID`.
* **Feature Engineering:** Implements Binary Encoding and One-Hot Encoding for categorical variables.
* **High-Performance Export:** Saves processed data in **Parquet** format to preserve data types and optimize speed [cite: 2026-02-01].

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


