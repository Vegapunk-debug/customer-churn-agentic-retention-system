import joblib
Cluster_descriptions = {
    0: "The Loyal Spenders: Long-time customers, high bills, 1-2 year contracts, rarely leave.",
    1: "The At-Risk Group: Newer customers, high bills, month-to-month plans, quitting fast.",
    2: "The Average Users: Basic internet users, average bills, normal cancellation rate.",
    3: "The Phone-Only Group: No internet, tiny bills, very stable, rarely leave.",
    4: "The Internet-Only Group: No phone service, just internet, normal cancellation rate."
}
def identify_user_cluster(user_df, model_path='../models/churn_kmeans_pipeline.pkl'):
    
    pipeline = joblib.load(model_path)
    
    cluster_id = pipeline.predict(user_df)[0]

    description = Cluster_descriptions.get(cluster_id, "Unknown Archetype")
    
    return cluster_id, description

# --- Usage ---
# id, desc = identify_user_cluster(new_user_dataframe)
# print(f"Assigned to {id}: {desc}")