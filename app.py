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
