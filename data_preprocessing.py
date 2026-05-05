import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Data loading from csv or excel file
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")


# Data summary(numeric or categorical)
def get_data_summary(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    numeric_summary = df[numeric_cols].describe().T if len(numeric_cols) > 0 else pd.DataFrame()
    categorical_summary = df[categorical_cols].describe().T if len(categorical_cols) > 0 else pd.DataFrame()
    
    return numeric_summary, categorical_summary, numeric_cols, categorical_cols


# Missing value detection and handling
def detect_missing_values(df):
    return df.columns[df.isnull().any()].tolist()

def handle_missing_values(df, missing_strategy):
    df_copy = df.copy()
    for col, (strategy, custom_value) in missing_strategy.items():
        if strategy == "Mean":
            imp = SimpleImputer(strategy='mean')
            df_copy[col] = imp.fit_transform(df_copy[[col]])
        elif strategy == "Median":
            imp = SimpleImputer(strategy='median')
            df_copy[col] = imp.fit_transform(df_copy[[col]])
        elif strategy == "Mode":
            imp = SimpleImputer(strategy='most_frequent')
            df_copy[col] = imp.fit_transform(df_copy[[col]])
        elif strategy == "Drop":
            df_copy.drop(columns=[col], inplace=True)
        elif strategy == "Custom":
            imp = SimpleImputer(strategy='constant', fill_value=custom_value)
            df_copy[col] = imp.fit_transform(df_copy[[col]])
    return df_copy


# Detect task type based on target variable
def detect_task_type(df, target_col):
    if pd.api.types.is_numeric_dtype(df[target_col]):
        return "Regression" if df[target_col].nunique() > 20 else "Classification"
    return "Classification"


# Encoding categorical features using LabelEncoder and target variable
def encode_categorical(df, columns = None):
    df_copy = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns
    
    le_dict = {}
    for col in columns:
        le = LabelEncoder()
        df_copy[col] = le.fit_transform(df_copy[col].astype(str))
        le_dict[col] = le
    return df_copy, le_dict


# prepare data for modeling (handle missing values, encode categorical features, split into features and target)
def prepare_data(df, feature_cols, target_col=None):
    X = df[feature_cols]
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        imp = SimpleImputer(strategy='mean')
        X[numeric_cols] = imp.fit_transform(X[numeric_cols])

    X = pd.get_dummies(X)
    label_encoder_info = {}
    y = None
    
    if target_col:
        y = df[target_col].copy()
        if y.isnull().any():
            if pd.api.types.is_numeric_dtype(y):
                imp = SimpleImputer(strategy='mean')
            else:
                imp = SimpleImputer(strategy='most_frequent')
            y = pd.Series(imp.fit_transform(y.values.reshape(-1, 1)).ravel(), index=y.index)
        
        # Encode categorical target variable
        if not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            label_encoder_info = {
                'encoder': le,
                'classes': le.classes_,
                'mapping': dict(zip(le.classes_, le.transform(le.classes_)))
            }
            y = pd.Series(y_encoded, index=y.index)
    
    return X, y, label_encoder_info 


# Apply normalization methods
def normalize_data(X, method):
    if method == "Minmax":
        scaler = MinMaxScaler()
    elif method == "Standard":
        scaler = StandardScaler()
    elif method == "Robust":
        scaler = RobustScaler()
    else:
        return X
    return pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# Split data into train and test sets
def split_and_scale(X, y=None, test_size=0.2, normalization_method=StandardScaler()):
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train = normalize_data(X_train, normalization_method)
        X_test = normalize_data(X_test, normalization_method)
        return X_train, X_test, y_train, y_test
    else:
        return normalize_data(X, normalization_method)


# Apply PCA for dimensionality reduction
def apply_pca(X, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X), pca