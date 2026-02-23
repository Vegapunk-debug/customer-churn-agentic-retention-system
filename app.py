import streamlit as st
import pandas as pd
import joblib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Retention System",
    page_icon="📊",
    layout="wide"
)

# --- Constants & Paths ---
MODEL_PATH = 'models/rf_model.pkl'

# --- Load Model ---
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

# --- Preprocessing Logic ---
def preprocess_raw_sample(raw_data, training_columns):
    """
    Applies the manual encoding steps used in training.
    """
    df_sample = pd.DataFrame([raw_data])
    
    # Binary Encoding
    binary_map = {'Yes': 1, 'No': 0}
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df_sample.columns:
            df_sample[col] = df_sample[col].replace(binary_map)
    
    if 'gender' in df_sample.columns:
        df_sample['gender'] = df_sample['gender'].replace({'Female': 1, 'Male': 0})
        
    # TotalCharges numeric conversion
    if 'TotalCharges' in df_sample.columns:
        df_sample['TotalCharges'] = pd.to_numeric(df_sample['TotalCharges'], errors='coerce')
    
    # One-Hot Encoding
    multi_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaymentMethod'
    ]
    
    df_sample = pd.get_dummies(df_sample, columns=[c for c in multi_cols if c in df_sample.columns], dtype=int)
    
    # Align columns
    for col in training_columns:
        if col not in df_sample.columns:
            df_sample[col] = 0
            
    return df_sample[training_columns]

# --- UI ---
st.title("🛡️ Customer Churn Prediction System")
st.markdown("""
Input customer details below to predict the likelihood of churn. 
This system uses a **Random Forest Classifier** trained on historical telecom data.
""")

if model is None:
    st.error(f"Model file not found at `{MODEL_PATH}`. Please ensure the model is trained and saved.")
else:
    training_features = model.feature_names_in_

    with st.form("customer_form"):
        st.subheader("Customer Demographics & Services")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            
        with col2:
            tenure = st.slider("Tenure (Months)", 0, 72, 12)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

        with col3:
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

        st.divider()
        st.subheader("Contract & Billing")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        with col5:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

        with col6:
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=50.0)

        submit = st.form_submit_button("🔍 Predict Churn Risk")

    if submit:
        # Construct raw input dict
        raw_input = {
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': str(total_charges)
        }
        
        # Preprocess and Predict
        processed_sample = preprocess_raw_sample(raw_input, training_features)
        prediction = model.predict(processed_sample)[0]
        probability = model.predict_proba(processed_sample)[0][1]
        
        # Display Results
        st.divider()
        st.subheader("Prediction Result")
        
        if prediction == 1:
            st.error(f"🚨 **High Risk**: This customer is likely to CHURN. (Probability: {probability:.2%})")
            st.progress(probability)
        else:
            st.success(f"✅ **Low Risk**: This customer is likely to STAY. (Churn Probability: {probability:.2%})")
            st.progress(probability)
        
        with st.expander("Show Technical Details"):
            st.write("Processed Input Vector:")
            st.dataframe(processed_sample)
