from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import numpy as np


def calculate_metrics(y_test, y_pred, task_type):
    if task_type == "Regression":
        return {
            "MSE": mean_squared_error(y_test, y_pred),
            "R2 Score": r2_score(y_test, y_pred)
        }
    else:
        return {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Classification Report": classification_report(y_test, y_pred, output_dict=True),
            "Confusion Matrix": confusion_matrix(y_test, y_pred)
        }


def decode_predictions(y_pred, label_encoder_info):
    """
    Convert encoded predictions back to original class names
    """
    if label_encoder_info and 'encoder' in label_encoder_info:
        return label_encoder_info['encoder'].inverse_transform(y_pred)
    return y_pred