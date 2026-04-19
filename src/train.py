import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    roc_auc_score
)

def train_model():
    print("Loading data...")
    # Read the processed parquet data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed_churn_data.parquet")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at {data_path}. Please run preprocessing first.")
        
    df = pd.read_parquet(data_path)

    # Separate features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y   
    )

    # Initialize Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    )
    
    # ---------------------------------------------------------
    # Perform Cross-Validation to demonstrate robustness
    # ---------------------------------------------------------
    print("Running Cross-Validation on the training set...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='accuracy')
    
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print("-" * 50)

    # ---------------------------------------------------------
    # Train the final model on the full training set
    # ---------------------------------------------------------
    print("Training the final Random Forest model...")
    rf_model.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = rf_model.predict(X_test)
    y_probs = rf_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_probs)

    print("\n--- Evaluation Metrics on Test Set ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC Score: {roc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ---------------------------------------------------------
    # Save the Model Artifact
    # ---------------------------------------------------------
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "rf_model.pkl")
    joblib.dump(rf_model, model_path)
    print(f"\nModel strictly saved to: {model_path}")

if __name__ == "__main__":
    train_model()
