import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from data_preprocessing import encode_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_correlation_heatmap(df, feature_cols, target_col, st):
    corr_cols = feature_cols + [target_col]
    df_subset = df[corr_cols]

    categorical_cols = df_subset.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        df_encoded, _ = encode_categorical(df_subset, categorical_cols)
    else:
        df_encoded = df_subset

    corr_df = df_encoded.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', ax=ax)
    plt.title("Correlation Heatmap")
    st.pyplot(fig)
        

def plot_regression_results(y_test, y_pred, st):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted")
    st.pyplot(fig)

def plot_classification_results(y_test, y_pred, st):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax)
    plt.title("Confusion Matrix")
    st.pyplot(fig)

def plot_feature_importance(model, feature_cols, st):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importance})
        importance_df.sort_values(by='Importance', ascending=False, inplace=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
        plt.title("Feature Importance")
        st.pyplot(fig)


def plot_model_comparison(results, task_type, st):
    if task_type == "Regression":
        scores = {name: result['score'] for name, result in results.items()}
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=list(scores.keys()), y=list(scores.values()), ax=ax)
        plt.ylabel("R2 Score")
        plt.title("Model Comparison - R2 Score")
        st.pyplot(fig)
    else:
        scores = {name: result['score'] for name, result in results.items()}
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=list(scores.keys()), y=list(scores.values()), ax=ax)
        plt.ylabel("Accuracy")
        plt.title("Model Comparison - Accuracy")
        st.pyplot(fig)

def plot_clusters(df, cluster_labels, st):
    df['Cluster'] = cluster_labels
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=df.index, y=df['Cluster'], hue=df['Cluster'], palette='Set2', ax=ax)
    plt.title("Cluster Assignments")
    plt.xlabel("Data Point Index")
    plt.ylabel("Cluster Label")
    st.pyplot(fig)