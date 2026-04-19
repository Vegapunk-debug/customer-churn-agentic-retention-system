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
- **Inference & Agentic UI:** `Streamlit`, `Shap`, `KaggleHub`

## Milestone 2: Model Training, Evaluation & Retention Placeholder

The second phase introduces a formal training pipeline, robust evaluation, and the foundation for agentic retention strategies.

### Key Features
* **Dedicated Training Pipeline (`src/train.py`):** A standalone script that handles data loading, preprocessing, model training, and saving artifacts.
* **Robust Evaluation:** Implements 5-fold cross-validation to ensure model stability and reports Accuracy, ROC-AUC, and Confusion Matrix.
* **Retention Automation Layer (`src/retention_automation.py`):** A modular placeholder for the upcoming agentic strategy engine, including placeholder logic for personalized outreach.
* **Comprehensive Testing:** Unit tests for both preprocessing cleaning steps and the retention module.

## Project Structure
```text
├── data/               # Raw and processed datasets
├── models/             # Saved model artifacts (.pkl) and results
├── notebooks/          # EDA and experiment notebooks
├── src/                # Source code
│   ├── preprocessing.py # Data cleaning and engineering
│   ├── train.py         # Robust training script
│   ├── inference.py     # SHAP explanations & prediction logic
│   └── retention_automation.py # Agentic placeholder
├── tests/              # Unit tests for core modules
├── app.py              # Streamlit dashboard
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

1. **Preprocessing:** Use the `notebooks/preprocessing.ipynb` to download and prepare the initial dataset.
2. **Training:** Run the training script to generate the model and evaluation metrics:
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   python3 src/train.py
   ```
3. **Inference / App:** Launch the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```

## Unit Testing

To ensure data integrity and reliability, run the unit tests:
```bash
python3 -m unittest discover tests
```

## Evaluation Results

After running `src/train.py`, metrics are saved to `models/evaluation_results.json`. 

| Metric | Value (Baseline) |
| :--- | :--- |
| **Accuracy** | ~80% |
| **ROC-AUC** | ~84% |
| **Cross-Validation Accuracy** | ~79% (+/- 1%) |

## Future Roadmap: Agentic Retention
The `src/retention_automation.py` module is ready for integration. Future updates will include:
- [ ] LLM-powered personalized retention message generation.
- [ ] Automated discount/coupon assignment based on churn risk.
- [ ] Integration with CRM tools for automated outreach.


