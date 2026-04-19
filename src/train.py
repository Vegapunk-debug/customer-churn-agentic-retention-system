import pandas as pd
import numpy as np
import os
import joblib
import json
import kagglehub
import shutil
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from src.preprocessing import preprocess_full_dataset

def download_data(local_path="data/raw_churn_data.csv"):
    """Downloads the Telco Churn dataset from Kaggle if not found."""
    print("Checking for dataset...")
    if not os.path.exists(local_path):
        print("Dataset not found locally. Downloading from Kaggle via kagglehub...")
        try:
            path = kagglehub.dataset_download("blastchar/telco-customer-churn")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            # Find the CSV file in the downloaded path
            csv_file = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
            src_file = os.path.join(path, csv_file)
            shutil.copy(src_file, local_path)
            print(f"Dataset downloaded and saved to {local_path}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False
    else:
        print("Dataset found locally.")
    return True

def train_model(data_path="data/raw_churn_data.csv", model_save_path="models/rf_model.pkl"):
    # Ensure data is present
    if not download_data(data_path):
        return

    print(f"Loading data from {data_path}...")

    df = pd.read_csv(data_path)
    
    print("Preprocessing data...")
    X, y = preprocess_full_dataset(df)
    
    # Ensure no NaN values remain
    X = X.fillna(0)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training Random Forest model with {X_train.shape[0]} samples...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    
    # Robustness check with Cross-Validation
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
    print(f"Cross-validation accuracy scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
    
    # Final training
    rf.fit(X_train, y_train)
    
    # Evaluation
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    results = {
        "accuracy": acc,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "cv_mean_accuracy": cv_scores.mean(),
        "cv_std_accuracy": cv_scores.std()
    }
    
    print("\nEvaluation Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model artifact
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(rf, model_save_path)
    print(f"\nModel saved to {model_save_path}")
    
    # Save metrics for documentation
    with open("models/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    return results

if __name__ == "__main__":
    train_results = train_model()
