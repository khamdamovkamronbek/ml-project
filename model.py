from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
import xgboost as xgb   
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix, silhouette_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import streamlit as st

def get_supervised_models(task_type):
    if task_type == "Regression":
        return {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(),
            "XGBoost Regressor": xgb.XGBRegressor()
        }
    else:
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest Classifier": RandomForestClassifier(),
            "XGBoost Classifier": xgb.XGBClassifier()
        }
    

def train_supervised(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    return model.predict(X_test)

def train_multiple_models(models, X_train, y_train, X_test, y_test, task_type):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if task_type == "Regression":
            score = r2_score(y_test, y_pred)
        else:
            score = accuracy_score(y_test, y_pred)
        results[name] = {'y_pred': y_pred, 'score': score}
    return results

def get_unsupervised_models():
    return {
        "KMeans": KMeans,
        "DBSCAN": DBSCAN
    }

def train_unsupervised(model, X_scaled, params):
    if isinstance(model, type(KMeans)):
        cluster_model = model(n_clusters=params['n_clusters'], random_state=42)
    elif isinstance(model, type(DBSCAN)):
        cluster_model = model(eps=params['eps'], min_samples=params['min_samples'])
    return cluster_model.fit_predict(X_scaled)


def get_hyperparameter_grid(model_name, task_type):
    """Returns hyperparameter search space for each model"""
    grids = {
        "Regression": {
            "Linear Regression": {},
            "Random Forest Regressor": {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            "XGBoost Regressor": {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 0.9, 1.0]
            }
        },
        "Classification": {
            "Logistic Regression": {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'penalty': ['l2'],
                'max_iter': [1000]
            },
            "Random Forest Classifier": {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            "XGBoost Classifier": {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
    }
    return grids[task_type].get(model_name, {})


def tune_hyperparameters(model, model_name, X_train, y_train, X_test, y_test, task_type, search_type="grid", n_iter=10):
    """
    Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV
    
    Parameters:
    - search_type: "grid" for GridSearchCV, "random" for RandomizedSearchCV
    - n_iter: number of iterations for RandomizedSearchCV
    """
    param_grid = get_hyperparameter_grid(model_name, task_type)
    
    if not param_grid:
        st.warning(f"No hyperparameters to tune for {model_name}")
        return model, None
    
    # Determine scoring metric
    scoring = 'r2' if task_type == "Regression" else 'accuracy'
    
    if search_type == "random":
        search = RandomizedSearchCV(model, param_grid, n_iter=n_iter, cv=5, 
                                   scoring=scoring, random_state=42, n_jobs=-1)
    else:
        search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1)
    
    search.fit(X_train, y_train)
    
    st.success(f"Best parameters found!")
    st.write(f"Best {scoring} score: {search.best_score_:.4f}")
    st.write(f"Best parameters: {search.best_params_}")
    
    # Test the best model
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    if task_type == "Regression":
        test_score = r2_score(y_test, y_pred)
    else:
        test_score = accuracy_score(y_test, y_pred)
    
    st.write(f"Test set score: {test_score:.4f}")
    
    return best_model, {'best_params': search.best_params_, 'best_cv_score': search.best_score_, 'test_score': test_score}


def get_unsupervised_hyperparameter_grid(model_name):
    """Returns hyperparameter search space for unsupervised models"""
    grids = {
        "KMeans": {
            'n_clusters': [2, 3, 4, 5, 6, 7, 8],
            'init': ['k-means++', 'random'],
            'n_init': [10, 20]
        },
        "DBSCAN": {
            'eps': [0.3, 0.5, 0.7, 1.0, 1.5],
            'min_samples': [2, 3, 5, 10]
        }
    }
    return grids.get(model_name, {})


def tune_unsupervised_hyperparameters(model_class, model_name, X_scaled, search_type="grid", n_iter=10):
    """
    Perform hyperparameter tuning for unsupervised models using silhouette score
    """
    param_grid = get_unsupervised_hyperparameter_grid(model_name)
    
    if not param_grid:
        st.warning(f"No hyperparameters to tune for {model_name}")
        return model_class(), None
    
    best_score = -1
    best_params = {}
    best_model = None
    
    if search_type == "random":
        import random
        param_combinations = []
        for _ in range(n_iter):
            combo = {}
            for param, values in param_grid.items():
                combo[param] = random.choice(values)
            param_combinations.append(combo)
    else:
        # Grid search - generate all combinations
        import itertools
        param_names = list(param_grid.keys())
        param_combinations = [dict(zip(param_names, values)) 
                             for values in itertools.product(*param_grid.values())]
    
    st.info(f"Testing {len(param_combinations)} parameter combinations...")
    progress_bar = st.progress(0)
    
    for idx, params in enumerate(param_combinations):
        try:
            if model_name == "KMeans":
                model = model_class(random_state=42, **params)
            else:  # DBSCAN
                model = model_class(**params)
            
            labels = model.fit_predict(X_scaled)
            
            # Check if we have at least 2 clusters and not all noise points
            if len(set(labels)) > 1 and len(set(labels)) < len(X_scaled):
                score = silhouette_score(X_scaled, labels)
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_model = model
        except Exception as e:
            continue
        
        progress_bar.progress((idx + 1) / len(param_combinations))
    
    if best_model is None:
        st.warning(f"Could not find valid clustering with given parameters")
        best_model = model_class(**param_grid)
        return best_model, None
    
    st.success(f"Best parameters found!")
    st.write(f"Best silhouette score: {best_score:.4f}")
    st.write(f"Best parameters: {best_params}")
    
    return best_model, {'best_params': best_params, 'silhouette_score': best_score}