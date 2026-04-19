# 📊 OUTLIER.AI | Customer Churn Prediction & Retention System

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/Agent-LangChain-green)](https://www.langchain.com/)
[![LLM-Groq](https://img.shields.io/badge/LLM-Groq-violet)](https://groq.com/)
[![Status: Functional](https://img.shields.io/badge/Status-Functional-brightgreen.svg)]()

**OUTLIER AI** is an advanced, agentic system designed to not only predict customer churn but also autonomously reason about risk factors and generate tailored retention strategies using Retrieval-Augmented Generation (RAG).

## 🚀 Key Features

- **Predictive Intelligence:** High-accuracy Random Forest classifier with 82.7% ROC-AUC.
- **Agentic Reasoning:** A multi-stage AI agent that analyzes SHAP-based churn drivers to understand *why* a customer is at risk.
- **RAG-Powered Strategies:** Pulls from a curated Tactical Knowledge Base to recommend industry-standard retention plays.
- **Dual Flow Architecture:** 
  - **Advanced Agent:** Structured strategy reports with reasoning logs and business disclaimers.
  - **Llama-3 Integration:** Deep synthesis using Llama-3-70b (via Groq) for conversational strategy generation.
- **Cyberpunk UI:** High-fidelity, glassmorphic dashboard built with Streamlit.

## 🛠 Tech Stack

- **Core AI:** `Scikit-Learn`, `SHAP`, `RandomForest`
- **Agentic Layer:** `LangChain`, `Groq (Llama-3)`, `FAISS`
- **Dashboard:** `Streamlit`, `Python-Dotenv`
- **Embedding Models:** `HuggingFace (all-MiniLM-L6-v2)`

## 📂 Project Structure

```text
├── models/             # Saved ML model artifacts (rf_model.pkl, etc.)
├── src/                # Core Logic
│   ├── retention_automation.py # Primary Agentic Retention logic
│   ├── retention_agent.py      # Upstream LLM/RAG integration
│   ├── inference.py            # SHAP & Logic orchestration
│   ├── preprocessing.py        # Data cleaning pipeline
│   └── train.py                # Automated training w/ 5-fold CV
├── app.py              # Main "Nexus" Dashboard entrypoint
├── requirements.txt    # Integrated dependencies (LangChain + ML)
├── .env                # API Key configuration
└── README.md
```

## 🏁 Setup & Installation

### 1. Environment Configuration
```bash
git clone https://github.com/Vegapunk-debug/customer-churn-agentic-retention-system.git
cd customer-churn-agentic-retention-system
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional but Recommended)
Create a `.env` file in the root directory:
```bash
GROQ_API_KEY=your_groq_api_key_here
```
*Get your key at [console.groq.com](https://console.groq.com/)*

### 3. Run the System
```bash
# To train/validate the model
python src/train.py

# To launch the dashboard
streamlit run app.py
```

## 📈 Model Insights
Our training pipeline uses **5-fold cross-validation** to ensure robustness:
- **Accuracy:** ~78.8%
- **ROC-AUC Score:** 82.7%
- **Top Feature Contributions:** `Contract_Month-to-month`, `TotalCharges`, `Tenure`.

## 📜 Deployment Note
When deploying to **Streamlit Cloud**, do not use a `.env` file. Instead, add your `GROQ_API_KEY` to the **Secrets** tab in the Streamlit Cloud Settings.


