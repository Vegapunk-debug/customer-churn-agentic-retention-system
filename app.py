import streamlit as st
import pandas as pd
import os
from src.preprocessing import preprocess_user_query
from src.inference import (
    load_rf_model, 
    random_forest_inference, 
    identify_user_cluster, 
    rf_feature_contribution_to_churn,
    display_prediction_results
)


st.set_page_config(
    page_title="Customer Retention System",
    page_icon="",
    layout="wide"
)


model = load_rf_model()


st.title(" Customer Churn Prediction System")
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

        submit = st.form_submit_button("Predict Churn Risk")

    if submit:

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
        

        processed_sample = preprocess_user_query(raw_input, training_features)
        

        prediction, probability = random_forest_inference(processed_sample)
        

        cluster_id, cluster_desc = identify_user_cluster(processed_sample)
        

        st.divider()
        
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            st.subheader("Prediction Result")
            display_prediction_results(prediction, probability)

        with col_res2:
            st.subheader("Customer Archetype")
            st.info(f"**Group {cluster_id}**: {cluster_desc}")


        st.divider()
        st.subheader("Risk Factor Analysis")
        st.markdown("This chart shows which factors contributed most to the prediction. Red bars increase churn risk, blue bars decrease it.")
        
        with st.spinner("Generating explanation..."):
            fig = rf_feature_contribution_to_churn(processed_sample)
            st.pyplot(fig)
        
        with st.expander("Show Technical Details"):
            st.write("Processed Input Vector:")
            st.dataframe(processed_sample)
