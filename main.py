import pandas as pd
import numpy as np
import streamlit as st
from utils import calculate_metrics
import matplotlib.pyplot as plt
from data_preprocessing import (apply_pca, encode_categorical, load_data, get_data_summary, detect_missing_values, 
                                handle_missing_values, detect_task_type, prepare_data, split_and_scale)
from visualization import (plot_clusters, plot_correlation_heatmap, plot_regression_results, plot_classification_results, plot_feature_importance, plot_model_comparison)
from model import (get_supervised_models, get_unsupervised_models, train_multiple_models, train_supervised, train_unsupervised, tune_hyperparameters, tune_unsupervised_hyperparameters)

# Streamlit app code goes here
st.set_page_config(page_title="ML App", layout="wide")
st.title("Machine Learning App")
st.write("Upload your datasets and this app allows you to explore and visualize machine learning models.")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file or Excel file", type=["csv", "xlsx"])
if uploaded_file is not None:
    df = load_data(uploaded_file)

    st.write("Dataset Preview:")
    st.dataframe(df.head())

    # Data summary
    st.write("Dataset Summary:")
    numeric_summary, categorical_summary, numeric_cols, categorical_cols = get_data_summary(df)
    st.write("Numeric Summary:")
    st.dataframe(numeric_summary)
    st.write("Categorical Summary:")
    st.dataframe(categorical_summary)

    # Handle missing values
    st.write("Handling Missing Values:")
    missing_columns = detect_missing_values(df)
    if missing_columns:
        st.write(f"Columns with missing values: {', '.join(missing_columns)}")
        
        # impute missing values
        missing_strategy = {}
        for col in missing_columns:
            strategy = st.selectbox(f"Select imputation strategy for {col}", ["Mean", "Median", 
                                                                              "Mode", "Drop", "Custom", "None"])
            custom_value = None
            if strategy == "Custom":
                custom_value = st.text_input(f"Enter custom value for {col}", key=f"custom_{col}")
                if col in numeric_cols and custom_value:
                    try:
                        custom_value = float(custom_value)  # convert to number
                    except ValueError:
                        st.error(f"{col} requires a numeric value!")
                        custom_value = None
            
            missing_strategy[col] = (strategy, custom_value)
        if st.button("Apply Imputation"):
            df = handle_missing_values(df, missing_strategy)
            st.write("Missing values handled. Updated dataset:")
            st.dataframe(df.head())
    else:
        st.write("No missing values found.")

    # Normalization options 
    st.write("Normalization:")
    normalization_method = st.selectbox("Select normalization method",
                                         ["None", "Min-Max Scaling", "Standardization", "Robust Scaling"],index=0)
    norm_dict ={
        "Min-Max Scaling": "Minmax",
        "Standardization": "Standard",
        "Robust Scaling": "Robust",
        "None": None
    }

    learning_type = st.selectbox("Select learning type", ["Supervised", "Unsupervised"])
    columns = df.columns.tolist()
    if learning_type == "Supervised":
        target_col = st.selectbox("Select target column", columns)
        feature_cols = [col for col in columns if col != target_col]
        detected_task = detect_task_type(df, target_col)
        st.write(f"Detected task type: {detected_task}")
         
        task_type = st.radio("Select task type", ["Classification", "Regression"],
                              index=0 if detected_task == "Regression" else 1)
        feature_cols = st.multiselect("Select feature columns", feature_cols, default=feature_cols)

        # correleation heatmap
        if st.checkbox("Show correlation heatmap"):
            st.subheader("Correlation Heatmap")
            plot_correlation_heatmap(df, feature_cols, target_col, st)
        
        # test_train_split
        test_size = st.slider("Test set size (%)", 10, 50, 20)
        st.write(f"Selected test set size: {test_size}%")
        model_options = get_supervised_models(task_type)
        model_choice = st.multiselect("Select a model", list(model_options.keys()), default=list(model_options.keys())[0])
        st.write(f"Selected model: {model_choice}")

        train_mode = st.radio("Train Model", ["Train Single Model", "Compare Multiple Models"], index=0)
        
        # Hyperparameter Tuning Options
        st.subheader("Hyperparameter Tuning")
        enable_tuning = st.checkbox("Enable Hyperparameter Tuning")
        tuning_config = {}
        if enable_tuning:
            tuning_config['search_type'] = st.radio("Search Type", ["grid", "random"], index=0)
            if tuning_config['search_type'] == "random":
                tuning_config['n_iter'] = st.slider("Number of iterations (Random Search)", 5, 50, 10)
            st.info("Tuning will use 5-fold cross-validation and may take some time...")
        
        if st.button("Train Model"):
            try:
                X, y, label_encoder_info = prepare_data(df, feature_cols, target_col)
                X_train, X_test, y_train, y_test = split_and_scale(X, y, test_size, 
                                                                   norm_dict[normalization_method])
                
                # Display label encoding info if target was categorical
                if label_encoder_info:
                    st.info(f"Target variable encoding: {label_encoder_info['mapping']}")

                if train_mode == "Train Single Model":
                    if len(model_choice) != 1:
                        st.error("Please select exactly one model to train.")
                    else:
                        model = model_options[model_choice[0]]
                        st.write(f"Training {model_choice[0]}...")
                        
                        if enable_tuning:
                            st.info("Tuning hyperparameters...")
                            from model import tune_hyperparameters
                            model, tuning_results = tune_hyperparameters(model, model_choice[0], X_train, y_train, 
                                                                        X_test, y_test, task_type, 
                                                                        search_type=tuning_config['search_type'],
                                                                        n_iter=tuning_config.get('n_iter', 10))
                            y_pred = model.predict(X_test)
                        else:
                            y_pred = train_supervised(model, X_train, y_train, X_test)

                        st.subheader("Model Performance")
                        plot_feature_importance(model, feature_cols, st)
                        metrics = calculate_metrics(y_test, y_pred, task_type)

                        for metric_name, metric_value in metrics.items():
                            if isinstance(metric_value, (int, float)):
                                st.write(f"{metric_name}: {metric_value:.4f}")
                            else:
                                st.write(f"{metric_name}:")
                                st.write(metric_value)
                        if task_type == "Regression":
                            plot_regression_results(y_test, y_pred, st) 
                        else:
                            plot_classification_results(y_test, y_pred, st)
                else:
                    if not model_choice:
                        st.error("Please select at least one model to compare.")
                    else:
                        models_to_train = {name: model_options[name] for name in model_choice}
                        results = train_multiple_models(models_to_train, X_train, y_train, X_test, y_test, task_type)
                    
                    
                    
                    st.write("Comparing multiple models...")
                    plot_model_comparison(results, task_type, st)

                    for name, result in results.items():
                        st.write(f"Model: {name}")
                        st.write(f"Score: {result['score']:.4f}")
                        if task_type == "Regression":
                            plot_regression_results(y_test, result['y_pred'], st)
                        else:
                            plot_classification_results(y_test, result['y_pred'], st)   
                    # Add code to compare selected models and display results
            except Exception as e:
                st.error(f"Error occurred while preparing data: {e}")

# Unsupervised learning
    else:
        st.write("Unsupervised learning functionality coming soon!")
        feature_cols = st.multiselect("Select feature columns for clustering", columns, default=columns)
        use_pca = st.checkbox("Apply PCA for dimensionality reduction")
        if use_pca:
            n_components = st.slider("Number of PCA components", 2, min(len(feature_cols), 10), 2)
        model_options = get_unsupervised_models()
        selected_model = st.selectbox("Select clustering algorithm", list(model_options.keys()))

        if selected_model == "KMeans":
            n_clusters = st.slider("Number of clusters (KMeans)", 2, 10, 3)
            params = {'n_clusters': n_clusters}
        else:
            eps = st.slider("Epsilon (DBSCAN)", 0.1, 5.0, 0.5)
            min_samples = st.slider("Min Samples (DBSCAN)", 2, 10, 5)
            params = {'eps': eps, 'min_samples': min_samples}

        # Hyperparameter Tuning for Unsupervised Models
        st.subheader("Hyperparameter Tuning")
        enable_clustering_tuning = st.checkbox("Enable Hyperparameter Tuning for Clustering")
        clustering_tuning_config = {}
        if enable_clustering_tuning:
            clustering_tuning_config['search_type'] = st.radio("Search Type", ["grid", "random"], index=0, key="clustering_search")
            if clustering_tuning_config['search_type'] == "random":
                clustering_tuning_config['n_iter'] = st.slider("Number of iterations (Random Search)", 5, 50, 10, key="clustering_iter")
            st.info("Tuning will evaluate multiple parameter combinations...")

        if st.button("Train Clustering Model"):
            try:
                df_encoded, _ = encode_categorical(df, feature_cols) if categorical_cols.any() else (df, {})
                X, _, _ = prepare_data(df_encoded, feature_cols)
                X_scaled = split_and_scale(X, normalization_method=norm_dict[normalization_method])
                if use_pca:
                    X_scaled, pca_model = apply_pca(X_scaled, n_components)
                    st.write(f"Explained variance by PCA: {pca_model.explained_variance_ratio_}")
                
                model_class = model_options[selected_model]
                
                if enable_clustering_tuning:
                    st.info("Tuning hyperparameters for clustering...")
                    clustering_model, tuning_results = tune_unsupervised_hyperparameters(
                        model_class, selected_model, X_scaled,
                        search_type=clustering_tuning_config['search_type'],
                        n_iter=clustering_tuning_config.get('n_iter', 10)
                    )
                    cluster_labels = clustering_model.fit_predict(X_scaled)
                else:
                    cluster_labels = train_unsupervised(model_class, X_scaled, params)
                
                st.write("Clustering completed. Cluster labels assigned to data points.")

                st.subheader("Cluster Distribution")
                plot_clusters(df, cluster_labels, st)
                st.write("Cluster counts:")
                st.write(pd.Series(cluster_labels).value_counts())
                if -1 in cluster_labels:
                    st.write("Note: -1 indicates noise points in DBSCAN.")
            except Exception as e:
                st.error(f"Error occurred while training unsupervised model: {e}")

st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload a CSV or Excel file containing your dataset.
2. Select the type of machine learning task (Supervised or Unsupervised).
3. For supervised learning, select the target column and choose models to train.
4. For unsupervised learning, select feature columns and choose a clustering algorithm.
""")

st.sidebar.header("Requirements")
st.sidebar.write("""
- Python 3.7+
- pandas
- scikit-learn
- streamlit
- seaborn
- matplotlib
""")