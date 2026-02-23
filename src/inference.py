import joblib
import shap
import matplotlib.pyplot as plt
import streamlit as st

Cluster_descriptions = {
    0: "The Loyal Spenders: Long-time customers, high bills, 1-2 year contracts, rarely leave.",
    1: "The At-Risk Group: Newer customers, high bills, month-to-month plans, quitting fast.",
    2: "The Average Users: Basic internet users, average bills, normal cancellation rate.",
    3: "The Phone-Only Group: No internet, tiny bills, very stable, rarely leave.",
    4: "The Internet-Only Group: No phone service, just internet, normal cancellation rate."
}


@st.cache_resource
def load_kmeans_pipeline(model_path='../models/kmeans_pipeline.pkl'):
    return joblib.load(model_path)

@st.cache_resource
def load_rf_model(model_path='../models/random_forest_model.pkl'):
    return joblib.load(model_path)

@st.cache_resource
def load_shap_explainer(_rf_model):
    return shap.TreeExplainer(_rf_model)


def identify_user_cluster(user_df):

    pipeline = load_kmeans_pipeline()
    cluster_id = pipeline.predict(user_df)[0]
    description = Cluster_descriptions.get(cluster_id, "Unknown Archetype")
    
    return cluster_id, description

def random_forest_inference(user_df):

    rf_model = load_rf_model()
    prediction = rf_model.predict(user_df)[0]
    return prediction

def rf_feature_contribution_to_churn(user_df):
    rf_model = load_rf_model()

    explainer = load_shap_explainer(rf_model)
    shap_values = explainer(user_df)

    if len(shap_values.shape) == 3:
        churn_explanation = shap_values[0, :, 1]
    else:
        churn_explanation = shap_values[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(churn_explanation, show=False)
    plt.tight_layout()
    
    return fig