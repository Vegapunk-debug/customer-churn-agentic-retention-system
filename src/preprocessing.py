import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data cleaning steps shared between training and inference."""
    df = df.copy()
    
    # Drop non-predictive columns
    df = df.drop(columns=['customerID', 'Churn'], errors='ignore')

    # Convert TotalCharges to numeric
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)

    # Binary encoding
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
            
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})
    
    # Fill remaining binary NAs
    cols_to_fix = [col for col in binary_cols + ['gender'] if col in df.columns]
    df[cols_to_fix] = df[cols_to_fix].fillna(0).astype(int)
    
    return df

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode multi-category columns."""
    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                  'Contract', 'PaymentMethod']
    
    cols_present = [col for col in multi_cols if col in df.columns]
    if cols_present:
        df = pd.get_dummies(df, columns=cols_present, dtype=int)
    
    return df

def preprocess_user_query(user_data: dict, expected_columns: list) -> pd.DataFrame:
    """
    Transforms raw user input from Streamlit app into a aligned feature vector.
    """
    df = pd.DataFrame([user_data])
    df = clean_data(df)
    df = encode_categorical(df)
    
    # Align columns with training data
    df = df.reindex(columns=expected_columns, fill_value=0)
    return df

def preprocess_full_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete preprocessing for the full dataset (training).
    """
    y = None
    if 'Churn' in df.columns:
        y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    X = clean_data(df)
    X = encode_categorical(X)
    
    return X, y