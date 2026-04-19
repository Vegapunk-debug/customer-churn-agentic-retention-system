import streamlit as st
from src.preprocessing import preprocess_user_query
from src.inference import (
    random_forest_inference, 
    identify_user_cluster, 
    rf_feature_contribution_to_churn,
    display_prediction_results
)

def handle_prediction(raw_input, training_features):
    processed_sample = preprocess_user_query(raw_input, training_features)
    st.session_state['processed_df'] = processed_sample
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
