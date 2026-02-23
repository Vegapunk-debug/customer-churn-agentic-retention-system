import pandas as pd

def preprocess_user_query(user_data: dict, expected_columns: list) -> pd.DataFrame:
    """
    The function transforms raw user input from a Streamlit app into a mathematically 
    aligned feature vector for Random Forest inference
    """
    df = pd.DataFrame([user_data])
    df = df.drop(columns=['customerID', 'Churn'], errors='ignore')

    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)


    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
            
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})
    
    
    cols_to_fix = [col for col in binary_cols + ['gender'] if col in df.columns]

    df[cols_to_fix] = df[cols_to_fix].fillna(0).astype(int)


    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                  'Contract', 'PaymentMethod']
    
    cols_present = [col for col in multi_cols if col in df.columns]

    if cols_present:
        df = pd.get_dummies(df, columns=cols_present, dtype=int)

    df = df.reindex(columns=expected_columns, fill_value=0)

    return df