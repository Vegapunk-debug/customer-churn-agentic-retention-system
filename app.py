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
    page_title="OUTLIER.AI | The Churn Prediction System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def inject_hyper_ai_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;500;700&family=Outfit:wght@200;400;600&display=swap');

        :root {
            --bg-deep: #050505;
            --accent-emerald: #00ff9f;
            --accent-amber: #ffb400;
            --accent-violet: #8b5cf6;
            --glass-white: rgba(255, 255, 255, 0.03);
            --border-glow: rgba(0, 255, 159, 0.15);
            --font-main: 'Outfit', sans-serif;
            --font-hdr: 'Space Grotesk', sans-serif;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stApp {
            background-color: var(--bg-deep);
            background-image: 
                linear-gradient(rgba(0, 255, 159, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 255, 159, 0.03) 1px, transparent 1px);
            background-size: 50px 50px;
            color: #ffffff;
        }

        ::-webkit-scrollbar {
            width: 5px;
        }
        ::-webkit-scrollbar-track {
            background: #050505;
        }
        ::-webkit-scrollbar-thumb {
            background: var(--accent-emerald);
            border-radius: 10px;
        }

        html, body, [class*="css"] {
            font-family: var(--font-main);
        }
        
        h1, h2, h3, .app-name {
            font-family: var(--font-hdr);
            letter-spacing: -0.01em;
        }

        .nexus-card {
            background: rgba(10, 10, 10, 0.8);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 255, 159, 0.1);
            border-left: 4px solid var(--accent-emerald);
            padding: 2.5rem;
            margin-bottom: 2rem;
            position: relative;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            transition: all 0.3s ease;
        }
        
        .nexus-card:hover {
            border-color: var(--accent-emerald);
            box-shadow: 0 0 20px rgba(0, 255, 159, 0.1);
            transform: translateY(-2px);
        }

        @keyframes revealUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes scanline {
            0% { transform: translateY(-100%); }
            100% { transform: translateY(100%); }
        }

        @keyframes glitch {
            0% { text-shadow: 2px 0 0 red, -2px 0 0 blue; }
            20% { text-shadow: -2px 0 0 red, 2px 0 0 blue; }
            40% { text-shadow: 2px 0 0 red, -2px 0 0 blue; }
            60% { text-shadow: -2px 0 0 red, 2px 0 0 blue; }
            80% { text-shadow: 2px 0 0 red, -2px 0 0 blue; }
            100% { text-shadow: -2px 0 0 red, 2px 0 0 blue; }
        }

        .glitch-text:hover {
            animation: glitch 0.3s infinite;
        }

        .reveal {
            animation: revealUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }

        .scanline {
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: linear-gradient(to bottom, transparent, rgba(0, 255, 159, 0.01) 50%, transparent);
            z-index: 9999;
            pointer-events: none;
            animation: scanline 8s linear infinite;
        }

        .nexus-header {
            text-align: left;
            padding: 6rem 0;
            margin-bottom: 4rem;
        }

        .nexus-title {
            font-size: 6rem;
            font-weight: 900;
            color: #fff;
            line-height: 0.8;
            letter-spacing: -5px;
            text-transform: uppercase;
        }
        
        .nexus-subtitle {
            color: var(--accent-emerald);
            font-family: var(--font-hdr);
            font-weight: 700;
            font-size: 0.9rem;
            letter-spacing: 0.5em;
            margin-top: 1.5rem;
            opacity: 0.8;
        }

        div[data-baseweb="select"] > div {
            background-color: rgba(255,255,255,0.02) !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            border-radius: 4px !important;
            transition: all 0.3s ease !important;
        }
        
        div[data-baseweb="select"] > div:hover {
            border-color: var(--accent-emerald) !important;
        }
        
        .stNumberInput input {
            background-color: rgba(255,255,255,0.02) !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            color: #fff !important;
        }

        div.stButton > button {
            width: 100%;
            background: var(--accent-emerald) !important;
            border: none !important;
            color: #000 !important;
            padding: 1rem !important;
            font-family: var(--font-hdr) !important;
            font-weight: 900 !important;
            text-transform: uppercase;
            letter-spacing: 2px;
            transition: all 0.3s ease !important;
        }
        
        div.stButton > button:hover {
            transform: scale(1.02);
            box-shadow: 0 0 30px rgba(0, 255, 159, 0.4);
        }

        .metric-value {
            font-size: 5rem;
            font-weight: 800;
            font-family: var(--font-hdr);
            line-height: 1;
        }
        
        .metric-label {
            font-size: 0.75rem;
            color: rgba(0, 255, 159, 0.6);
            text-transform: uppercase;
            letter-spacing: 0.3em;
            font-weight: 700;
        }

        @keyframes pulse {
            from { opacity: 0.3; transform: scaleY(0.5); }
            to { opacity: 1; transform: scaleY(1); }
        }
        </style>
        <div class="scanline"></div>
    
    """, unsafe_allow_html=True)

def render_nexus_header():
    st.markdown("""
        <div class="nexus-header reveal">
            <div class="nexus-title glitch-text">OUTLIER AI</div>
            <div class="nexus-subtitle">CHURN PREDICTION AND RETENTION SYSTEM // V1.0</div>
        </div>
    """, unsafe_allow_html=True)

def render_neural_metrics():
    st.markdown("""
        <div class="reveal" style="display: flex; justify-content: space-between; margin-bottom: 2rem; padding: 1rem; background: rgba(0,255,159,0.02); border: 1px solid rgba(0,255,159,0.1); font-family: 'Space Grotesk'; font-size: 0.7rem; color: rgba(0,255,159,0.5); letter-spacing: 0.1em;">
            <div>SYNC_STATUS: <span style="color: var(--accent-emerald);">OPTIMAL</span></div>
            <div>LATENCY: <span style="color: var(--accent-emerald);">14ms</span></div>
            <div>MODEL_IDENT: <span style="color: var(--accent-emerald);">RF_V1.8_ENCORE</span></div>
            <div>ACTIVE_VECTORS: <span style="color: var(--accent-emerald);">7,421</span></div>
            <div>SECURITY_ARMOR: <span style="color: var(--accent-emerald);">SHIELD_MAX</span></div>
        </div>
    """, unsafe_allow_html=True)

inject_hyper_ai_css()
render_nexus_header()
render_neural_metrics()



model = load_rf_model()


st.title(" Customer Churn Prediction System")
st.markdown("""
Input customer details below to predict the likelihood of churn. 
This system uses a **Random Forest Classifier** trained on historical telecom data.
""")

if model is None:
    st.error("Model file not found at `models/rf_model.pkl`. Please ensure the model is trained and saved.")
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
