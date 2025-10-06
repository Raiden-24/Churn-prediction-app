# Customer Churn Prediction Application

## Project Overview
This application predicts customer churn for a telecommunications company using machine learning. It features an interactive Streamlit interface that allows users to input customer data and receive predictions on whether a customer is likely to churn, along with visualizations of model performance and data insights.

## Features
- **Interactive Prediction Interface**: Input customer details and get real-time churn predictions
- **Advanced ML Model**: Improved Random Forest model with enhanced recall for churn detection
- **Feature Engineering**: Sophisticated feature transformations to improve prediction accuracy
- **Model Explainability**: SHAP values to explain individual predictions
- **Customer Segmentation**: Visualization of customer segments and their churn patterns
- **Data Exploration Dashboard**: In-depth analysis of customer data and churn factors
- **Retention Strategy Recommendations**: Tailored suggestions based on customer profiles

## Technical Details

### Machine Learning Models
- **Original Model**: Random Forest Classifier with basic preprocessing
- **Improved Model**: Enhanced Random Forest with:
  - Advanced feature engineering
  - Class imbalance handling using SMOTE
  - Hyperparameter tuning
  - Significantly improved recall for churn detection (72% vs 46%)

### Feature Engineering
- Tenure grouping
- Service usage aggregation
- Premium service flags
- Interaction features
- Charge difference analysis

### Technologies Used
- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning algorithms
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Data visualization
- **SHAP**: Model explainability
- **Imbalanced-learn**: Class imbalance handling

## Installation and Usage

### Prerequisites
- Python 3.7+
- pip package manager

### Setup
1. Clone this repository
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Run the application:
```
streamlit run app.py
```

### Using the Application
1. Input customer details in the sidebar
2. Click "Predict" to see churn prediction
3. Explore model performance and data insights in the other tabs

## Project Structure
- `app.py`: Main Streamlit application
- `model_training.ipynb`: Original model training notebook
- `improved_model_training.ipynb`: Enhanced model training notebook
- `improved_model.py`: Script for training the improved model
- `churn_model.pkl`: Original trained model
- `improved_churn_model.pkl`: Enhanced trained model
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: Dataset

## Model Performance
The improved model achieves:
- Accuracy: 77%
- Precision for churn: 55%
- Recall for churn: 72% (significant improvement from 46%)
- F1-score for churn: 0.62

## Future Improvements
- Integration with real-time customer data
- Automated retraining pipeline
- More sophisticated retention strategy recommendations
- A/B testing framework for retention strategies

## Skills Demonstrated
- Machine Learning
- Data Preprocessing and Feature Engineering
- Model Evaluation and Improvement
- Interactive Dashboard Development
- Business Application of ML
- Data Visualization
- Python Programming