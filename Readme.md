# Customer Churn Prediction - End-to-End ML Project
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-brightgreen?logo=streamlit)](https://sonalikasingh17-customer-churn-analysis-streamlit-app-ouocdt.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)](https://scikit-learn.org)


A complete machine learning project for predicting customer churn using multiple data sources including demographics, transaction history, customer service interactions, and online activity patterns.

## 🚀 Features

- **Complete ML Pipeline**: Data ingestion, transformation, model training, and prediction
- **Multiple Data Sources**: Demographics, transactions, customer service, online activity
- **Advanced Feature Engineering**: Transaction aggregations, behavioral patterns
- **Model Comparison**: Multiple algorithms with hyperparameter tuning
- **Interactive Web App**: Streamlit-based dashboard for predictions and analytics
- **Production Ready**: Modular code structure with proper error handling and logging

## 📊 Project Structure

```
customer-churn-prediction/
├── src/
│   ├── __init__.py
│   ├── exception.py          # Custom exception handling
│   ├── logger.py            # Logging configuration
│   ├── utils.py             # Utility functions
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py     # Data loading and merging
│   │   ├── data_transformation.py # Feature engineering
│   │   └── model_trainer.py      # Model training and evaluation
│   └── pipeline/
│       ├── __init__.py
│       ├── train_pipeline.py     # Training orchestration
│       └── predict_pipeline.py   # Prediction pipeline
├── artifacts/               # Generated models and data
├── logs/                   # Application logs
├── notebooks/              # Jupyter notebooks for analysis
├── streamlit_app.py        # Web application
├── requirements.txt        # Dependencies
├── setup.py               # Package setup
└── README.md
```

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd customer-churn-prediction
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install project as package**
```bash
pip install -e .
```

## 📈 Usage

### 1. Train the Model

```bash
python src/pipeline/train_pipeline.py
```

This will:
- Load and merge data from multiple Excel sheets
- Perform feature engineering and data preprocessing
- Train multiple ML models with hyperparameter tuning
- Save the best model and preprocessor

### 2. Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

Navigate to `http://localhost:8501` to access the web application.

### 3. Web Application Features

- **Prediction Page**: Input customer details and get churn probability
- **Analytics Dashboard**: Visualize data insights and churn patterns
- **About Page**: Project information and technical details

## 🔍 Data Sources

The project uses multiple data sources:

1. **Customer Demographics**: Age, gender, marital status, income level
2. **Transaction History**: Spending patterns, amounts, product categories
3. **Customer Service**: Interaction types, resolution status
4. **Online Activity**: Login frequency, service usage patterns
5. **Churn Status**: Target variable (0: Retained, 1: Churned)

## 🤖 Machine Learning Pipeline

### Data Processing
- **Feature Engineering**: Transaction aggregations, behavioral metrics
- **Preprocessing**: Scaling, encoding, imputation
- **Feature Selection**: Important features for churn prediction

### Model Training
- **Algorithms**: Random Forest, XGBoost, CatBoost, Logistic Regression, etc.
- **Hyperparameter Tuning**: Grid search with cross-validation
- **Evaluation**: ROC-AUC score, accuracy, precision, recall

### Model Selection
- Best model selection based on ROC-AUC score
- Model persistence using pickle/dill

## 📊 Key Features Engineered

- **Transaction Metrics**: Total spent, average transaction, frequency
- **Behavioral Patterns**: Login frequency, service usage preferences
- **Customer Service Metrics**: Resolution rates, interaction frequency
- **Category-wise Spending**: Books, clothing, electronics, furniture, groceries
- **Temporal Features**: Transaction period, recency

## 🚀 Deployment

### Local Deployment
The Streamlit app can be run locally using:
```bash
streamlit run streamlit_app.py
```

### Cloud Deployment Options
- **Streamlit Cloud**: Direct deployment from GitHub
- **Heroku**: Using Procfile and requirements.txt
- **AWS/GCP**: Container-based deployment
- **Docker**: Containerized deployment

## 📈 Model Performance

The system evaluates multiple models and selects the best performer based on:
- **ROC-AUC Score**: Primary metric for model selection
- **Accuracy**: Overall prediction accuracy
- **Precision/Recall**: Class-specific performance

## 🔧 Configuration

### Environment Variables
Create a `.env` file for configuration:
```env
MODEL_PATH=artifacts/model.pkl
PREPROCESSOR_PATH=artifacts/preprocessor.pkl
LOG_LEVEL=INFO
```

### Hyperparameters
Model hyperparameters can be modified in `src/components/model_trainer.py`

## 🧪 Testing

Run tests using:
```bash
python -m pytest tests/
```

## 📝 Logging

The application uses comprehensive logging:
- **Info Level**: Pipeline progress and model performance
- **Error Level**: Exception handling and debugging
- **Log Files**: Stored in `logs/` directory with timestamps


## 🙏 Acknowledgments

- **Lloyds Banking Group** for providing this virtual internship and project opportunity.  
  This experience gave me practical exposure to the **end-to-end machine learning pipeline**, from data cleaning and preprocessing to model training, hyperparameter tuning, and deployment using Streamlit.  
  Key learnings include:
  - Handling and integrating multiple heterogeneous datasets  
  - Feature engineering and preprocessing (scaling, encoding, temporal feature extraction)  
  - Model comparison, evaluation, and the importance of recall in imbalanced datasets  
  - Building an interactive analytics dashboard for business use-cases  

- Scikit-learn community for excellent ML tools  
- Streamlit team for the amazing web framework  
- Open source contributors for various libraries used  

**Built with ❤️ for better customer retention strategies**




