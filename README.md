# 🤖 Machine Learning App

A comprehensive **Streamlit-based web application** for exploring, preprocessing, and training machine learning models without writing a single line of code. Perfect for data scientists, analysts, and ML enthusiasts who want to quickly experiment with datasets and different algorithms.

---

## ✨ Features

- **📊 Data Exploration & Analysis**
  - Upload CSV or Excel files
  - View dataset previews and statistical summaries
  - Analyze numeric and categorical columns separately
  - Detect and handle missing values with multiple strategies

- **🔧 Data Preprocessing**
  - Missing value imputation (Mean, Median, Mode, Custom)
  - Categorical variable encoding
  - Feature scaling (Standard, MinMax, Robust)
  - Dimensionality reduction with PCA
  - Automatic train-test splitting

- **🎯 Supervised Learning**
  - **Regression Models**: Linear Regression, Random Forest Regressor, XGBoost Regressor
  - **Classification Models**: Logistic Regression, Random Forest Classifier, XGBoost Classifier
  - Train multiple models simultaneously and compare performance
  - Hyperparameter tuning with GridSearch and RandomizedSearch

- **🔍 Unsupervised Learning**
  - **Clustering Models**: KMeans, DBSCAN
  - Automatic task detection and model selection
  - Silhouette score evaluation for clustering quality

- **📈 Visualization & Insights**
  - Correlation heatmaps
  - Clustering visualizations
  - Regression and classification result plots
  - Feature importance analysis
  - Model comparison charts

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Installation

1. **Clone or download this repository**
   ```bash
   cd ml-project/ml-app
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # Using venv
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📖 Usage

### Running the Application

```bash
streamlit run main.py
```

The application will open in your browser at `http://localhost:8501`

### Workflow

1. **Upload Dataset**: Click "Choose a CSV file or Excel file" to upload your data
2. **Explore Data**: View dataset preview and statistical summaries
3. **Handle Missing Values**: Select imputation strategies for columns with missing data
4. **Detect Task Type**: The app automatically detects if it's a regression or classification problem
5. **Preprocess Data**: Encode categorical variables, scale features, and apply PCA if needed
6. **Train Models**: Choose supervised or unsupervised learning
7. **Evaluate Results**: View model performance metrics and visualizations
8. **Tune Hyperparameters**: Optimize model performance with hyperparameter tuning

---

## 📁 Project Structure

```
ml-app/
├── main.py                    # Main Streamlit application
├── data_preprocessing.py      # Data loading, cleaning, and preprocessing functions
├── model.py                   # Machine learning model definitions and training
├── visualization.py           # Plotting and visualization functions
├── utils.py                   # Utility functions and metric calculations
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── data/                      # (Optional) Place your datasets here
```

### Module Descriptions

- **main.py**: Main Streamlit interface - orchestrates the entire workflow
- **data_preprocessing.py**: Handles data loading, cleaning, imputation, encoding, and scaling
- **model.py**: Implements supervised and unsupervised model training and hyperparameter tuning
- **visualization.py**: Creates all charts and plots for data and model analysis
- **utils.py**: Helper functions for metric calculation and data validation

---

## 🛠️ Supported Models

### Supervised Learning

**Regression:**
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

**Classification:**
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

### Unsupervised Learning

- KMeans Clustering
- DBSCAN Clustering

---

## 📊 Input Data Format

The app accepts:
- **CSV files** (.csv)
- **Excel files** (.xlsx)

**Requirements:**
- First row should contain column headers
- Mix of numeric and categorical columns is supported
- Target column for supervised learning (auto-detected)

---

## 🎯 Quick Examples

### Example 1: Predicting House Prices (Regression)
1. Upload a CSV with house features and prices
2. Select "Regression" task type
3. Choose your target variable (price)
4. Train multiple models and compare R² scores

### Example 2: Customer Segmentation (Clustering)
1. Upload customer data (numeric features)
2. Select "Unsupervised Learning"
3. Choose KMeans or DBSCAN
4. Visualize customer clusters

### Example 3: Customer Churn Prediction (Classification)
1. Upload customer churn dataset
2. Select "Classification" task type
3. Train classifiers and view confusion matrices
4. Compare model accuracies

---

## ⚙️ Configuration & Customization

### Adjusting Default Parameters

Edit the respective Python files to modify:
- **PCA components**: Change in `data_preprocessing.py`
- **Model hyperparameters**: Modify in `model.py`
- **Train-test split ratio**: Adjust in data preprocessing functions
- **Scaling methods**: Update in preprocessing pipeline

### Adding New Models

To add a new model:
1. Import the model in `model.py`
2. Add it to the appropriate `get_*_models()` function
3. The app will automatically include it in the training pipeline

---

## 📊 Metrics & Evaluation

### Regression Metrics
- R² Score
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

### Classification Metrics
- Accuracy Score
- Precision, Recall, F1-Score
- Confusion Matrix
- Classification Report

### Clustering Metrics
- Silhouette Score
- Cluster visualization plots

---

## 🚨 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Import errors** | Run `pip install -r requirements.txt` again |
| **File upload fails** | Ensure file is CSV or Excel format |
| **Missing value errors** | Handle missing values before model training |
| **Slow performance** | Use PCA for dimensionality reduction |
| **Port already in use** | Run `streamlit run main.py --server.port 8502` |

---

## 🔧 Dependencies

See `requirements.txt` for complete list. Key packages:
- **streamlit** - Web application framework
- **pandas** - Data manipulation
- **scikit-learn** - Machine learning algorithms
- **xgboost** - Gradient boosting models
- **matplotlib** - Visualization
- **numpy** - Numerical computing

---

## 🤝 Contributing

Contributions are welcome! To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Submit a pull request

---

## 📝 License

This project is open source and available under the MIT License.

---

## 📧 Support

For issues, questions, or suggestions, please open an issue on GitHub or contact the project maintainer.

---

## 🎓 Learning Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Tutorials](https://scikit-learn.org/stable/user_guide.html)
- [Machine Learning Mastery](https://machinelearningmastery.com/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

---

**Happy Machine Learning! 🚀**
