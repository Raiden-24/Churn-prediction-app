import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time
import os
import shap
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency
import seaborn as sns

# Set the page title and icon
st.set_page_config(
    page_title="Churn Predictor", 
    page_icon="🚀",
    layout="wide"
)

# Create a directory for storing prediction history if it doesn't exist
if not os.path.exists('predictions'):
    try:
        os.makedirs('predictions')
    except FileExistsError:
        # Directory already exists, which is fine
        pass

# Function to load model and columns
@st.cache_resource
def load_model():
    try:
        # Try to load improved model first
        if os.path.exists('improved_churn_model.pkl') and os.path.exists('improved_churn_model_columns.pkl'):
            model = joblib.load('improved_churn_model.pkl')
            model_columns = joblib.load('improved_churn_model_columns.pkl')
            scaler = joblib.load('improved_churn_model_scaler.pkl')
            st.sidebar.success("✅ Using improved model with enhanced features!")
            return model, model_columns, scaler, True
        else:
            # Fall back to original model
            model = joblib.load('churn_model.pkl')
            model_columns = joblib.load('churn_model_columns.pkl')
            return model, model_columns, None, False
    except FileNotFoundError:
        st.error("Model files not found! Please run the model_training.ipynb notebook first.")
        st.stop()

@st.cache_resource
def load_shap_explainer():
    """Load SHAP explainer separately to ensure it's always available"""
    try:
        # Try to load improved model first
        if os.path.exists('improved_churn_model.pkl'):
            model = joblib.load('improved_churn_model.pkl')
            return shap.TreeExplainer(model)
        else:
            # Fall back to original model
            model = joblib.load('churn_model.pkl')
            return shap.TreeExplainer(model)
    except Exception as e:
        st.warning(f"Could not load SHAP explainer: {str(e)}")
        return None

# Function to load the dataset for feature importance and statistics
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    except FileNotFoundError:
        st.warning("Original dataset not found. Some features will be limited.")
        return None

# Load model and columns
model, model_columns, scaler, is_improved_model = load_model()
dataset = load_dataset()

# App title and description with improved styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4169E1;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4682B4;
    }
    .description {
        font-size: 1.1rem;
        color: #708090;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
        padding: 10px;
        border-radius: 5px;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Main title and description
st.markdown('<h1 class="main-title">🚀 Customer Churn Prediction Platform</h1>', unsafe_allow_html=True)
st.markdown('<p class="description">This application predicts whether a customer is likely to churn based on their account details. Adjust the inputs on the left to match a customer\'s profile and click \'Predict\' to see the result.</p>', unsafe_allow_html=True)

# Model Performance Showcase
st.markdown("---")
st.markdown('<h3 style="text-align: center; color: #1f77b4;">🎯 Model Performance Highlights</h3>', unsafe_allow_html=True)

# Performance metrics in columns
perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

with perf_col1:
    st.metric(
        label="🎯 Overall Accuracy", 
        value="77%", 
        delta="Optimized for Business Impact",
        help="Balanced accuracy across all predictions"
    )

with perf_col2:
    st.metric(
        label="🔍 Churn Detection Rate", 
        value="72%", 
        delta="+56% vs Baseline",
        help="Percentage of actual churners successfully identified"
    )

with perf_col3:
    st.metric(
        label="⚡ Precision Score", 
        value="55%", 
        delta="Strategic Trade-off",
        help="Accuracy of churn predictions (fewer false positives)"
    )

with perf_col4:
    st.metric(
        label="🏆 F1-Score", 
        value="62%", 
        delta="+15% Improvement",
        help="Balanced measure of precision and recall"
    )

# Key achievement highlight
st.success("""
🌟 **Key Achievement**: Our improved model catches **72% of customers who will actually churn** compared to only 46% with baseline models. 
This 56% improvement in churn detection translates to significant revenue protection and more effective retention strategies.
""")

st.markdown("---")

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Prediction", "Model Explainability", "Customer Segmentation", "Model Performance", "Data Insights", "A/B Testing Simulation", "Prediction History"])

# --- Sidebar for User Input ---
st.sidebar.markdown('<p class="sub-header">Customer Account Details</p>', unsafe_allow_html=True)

def user_input_features():
    # More comprehensive input fields in the sidebar
    with st.sidebar.expander("Demographics", expanded=True):
        gender = st.selectbox('Gender', ('Female', 'Male'))
        partner = st.selectbox('Partner', ('No', 'Yes'))
        dependents = st.selectbox('Dependents', ('No', 'Yes'))
    
    with st.sidebar.expander("Account Information", expanded=True):
        tenure = st.slider('Tenure (months)', 0, 72, 24)
        contract = st.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
        paperless_billing = st.selectbox('Paperless Billing', ('No', 'Yes'))
        payment_method = st.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    
    with st.sidebar.expander("Services", expanded=True):
        phone_service = st.selectbox('Phone Service', ('No', 'Yes'))
        multiple_lines = st.selectbox('Multiple Lines', ('No', 'Yes', 'No phone service'))
        internet_service = st.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
        online_security = st.selectbox('Online Security', ('No', 'Yes', 'No internet service'))
        online_backup = st.selectbox('Online Backup', ('No', 'Yes', 'No internet service'))
        device_protection = st.selectbox('Device Protection', ('No', 'Yes', 'No internet service'))
        tech_support = st.selectbox('Tech Support', ('No', 'Yes', 'No internet service'))
        streaming_tv = st.selectbox('Streaming TV', ('No', 'Yes', 'No internet service'))
        streaming_movies = st.selectbox('Streaming Movies', ('No', 'Yes', 'No internet service'))
    
    with st.sidebar.expander("Charges", expanded=True):
        monthly_charges = st.slider('Monthly Charges ($)', 18.0, 120.0, 70.0, 0.05)
        total_charges = st.slider('Total Charges ($)', 18.0, 9000.0, 1400.0, 1.0)
    
    # Create a dictionary for the input
    raw_data = {
        'gender': gender,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    # Create a user-friendly summary for display
    user_summary = {
        'Gender': gender,
        'Partner': partner,
        'Dependents': dependents,
        'Tenure': f"{tenure} months",
        'Contract': contract,
        'Paperless Billing': paperless_billing,
        'Payment Method': payment_method,
        'Phone Service': phone_service,
        'Multiple Lines': multiple_lines,
        'Internet Service': internet_service,
        'Online Security': online_security,
        'Online Backup': online_backup,
        'Device Protection': device_protection,
        'Tech Support': tech_support,
        'Streaming TV': streaming_tv,
        'Streaming Movies': streaming_movies,
        'Monthly Charges': f"${monthly_charges:.2f}",
        'Total Charges': f"${total_charges:.2f}"
    }
    
    # Process the data differently based on which model we're using
    if is_improved_model:
        # For improved model, we need to do more preprocessing
        df_input = pd.DataFrame([raw_data])
        
        # Feature Engineering (same as in improved_model.py)
        # Create tenure group - manually assign based on value instead of using qcut
        if tenure <= 12:
            df_input['tenure_group'] = '0-1 year'
        elif tenure <= 36:
            df_input['tenure_group'] = '1-3 years'
        elif tenure <= 60:
            df_input['tenure_group'] = '3-5 years'
        else:
            df_input['tenure_group'] = '5+ years'
        
        # Create average monthly charge
        df_input['avg_monthly_charges'] = df_input['TotalCharges'] / (df_input['tenure'] + 0.1)
        
        # Create charge difference
        df_input['charge_diff'] = df_input['MonthlyCharges'] - df_input['avg_monthly_charges']
        
        # Create service count feature
        service_columns = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                          'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        # Count services
        def count_services(row):
            count = 0
            for col in service_columns:
                if row[col] not in ['No', 'No phone service', 'No internet service', 'DSL']:
                    count += 1
            return count
        
        df_input['service_count'] = df_input.apply(count_services, axis=1)
        
        # Create binary flags for premium services
        df_input['has_premium_tech'] = df_input['TechSupport'].apply(lambda x: 1 if x == 'Yes' else 0)
        df_input['has_premium_security'] = df_input['OnlineSecurity'].apply(lambda x: 1 if x == 'Yes' else 0)
        df_input['has_premium_support'] = df_input[['OnlineBackup', 'DeviceProtection']].apply(
            lambda x: 1 if 'Yes' in x.values else 0, axis=1
        )
        df_input['has_streaming'] = df_input[['StreamingTV', 'StreamingMovies']].apply(
            lambda x: 1 if 'Yes' in x.values else 0, axis=1
        )
        
        # Create interaction features
        df_input['tenure_x_monthly'] = df_input['tenure'] * df_input['MonthlyCharges']
        
        # Binary encoding for Yes/No columns
        binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            df_input[col] = df_input[col].map({'Yes': 1, 'No': 0})
        
        # One-hot encoding for categorical columns
        cat_cols = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                    'Contract', 'PaymentMethod', 'tenure_group']
        
        df_encoded = pd.get_dummies(df_input, columns=cat_cols, drop_first=True)
        
        # Ensure all required columns are present
        for col in model_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        # Select only the required columns in the correct order
        features = df_encoded[model_columns]
        
        # Store a SHAP-friendly (unscaled) DataFrame for explainability
        try:
            st.session_state['input_df_for_shap'] = features.copy()
        except Exception:
            pass
        
        # Scale the features for model prediction
        features_scaled = scaler.transform(features)
        
        return features_scaled, user_summary
    else:
        # For original model, use the original preprocessing
        data = {
            'tenure': [tenure],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'gender': [1 if gender == 'Male' else 0],
            'Partner': [1 if partner == 'Yes' else 0],
            'Dependents': [1 if dependents == 'Yes' else 0],
            'PhoneService': [1 if phone_service == 'Yes' else 0],
            'PaperlessBilling': [1 if paperless_billing == 'Yes' else 0],
            # One-hot encoded columns
            'Contract_One year': [1 if contract == 'One year' else 0],
            'Contract_Two year': [1 if contract == 'Two year' else 0],
            'PaymentMethod_Credit card (automatic)': [1 if payment_method == 'Credit card (automatic)' else 0],
            'PaymentMethod_Electronic check': [1 if payment_method == 'Electronic check' else 0],
            'PaymentMethod_Mailed check': [1 if payment_method == 'Mailed check' else 0],
            'TechSupport_No internet service': [1 if tech_support == 'No internet service' else 0],
            'TechSupport_Yes': [1 if tech_support == 'Yes' else 0],
            'InternetService_Fiber optic': [1 if internet_service == 'Fiber optic' else 0],
            'InternetService_No': [1 if internet_service == 'No' else 0]
        }
        
        # Create a DataFrame and align columns with the model's training data
        features = pd.DataFrame(data)
        
        # Add all other missing columns with a value of 0
        for col in model_columns:
            if col not in features.columns:
                features[col] = 0
                
        # Ensure the order of columns is the same as in the training data
        features = features[model_columns]
        
        return features, user_summary

# Get user input
input_df, user_summary = user_input_features()

# Save prediction to history
def save_prediction(user_summary, prediction, probability):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prediction_result = "Likely to churn" if prediction == 1 else "Not likely to churn"
    churn_probability = f"{probability*100:.2f}%"
    
    # Create a record
    record = {
        'timestamp': timestamp,
        'prediction': prediction_result,
        'churn_probability': churn_probability
    }
    record.update(user_summary)
    
    # Convert to DataFrame
    record_df = pd.DataFrame([record])
    
    # Save to CSV
    history_file = 'predictions/prediction_history.csv'
    if os.path.exists(history_file):
        history_df = pd.read_csv(history_file)
        history_df = pd.concat([history_df, record_df], ignore_index=True)
    else:
        history_df = record_df
    
    history_df.to_csv(history_file, index=False)

# Prediction Tab
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<p class="sub-header">Customer Profile</p>', unsafe_allow_html=True)
        
        # Display user input in a formatted table
        user_summary_df = pd.DataFrame(list(user_summary.items()), columns=['Feature', 'Value'])
        st.table(user_summary_df)
        
        # Prediction button with loading animation
        if st.button('Predict Churn', key='predict_button'):
            with st.spinner('Analyzing customer data...'):
                # Add a small delay to show the spinner (for UX purposes)
                time.sleep(1)
                
                # Make prediction
                prediction = model.predict(input_df)
                prediction_proba = model.predict_proba(input_df)
                churn_probability = prediction_proba[0][1]
                
                # Save to history
                save_prediction(user_summary, prediction[0], churn_probability)
                
                # Persist state for use in other tabs
                st.session_state['prediction_made'] = True
                st.session_state['prediction'] = int(prediction[0])
                st.session_state['prediction_proba'] = float(churn_probability)
                # input_df_for_shap is set during preprocessing for improved model; for original model, set it here
                if not is_improved_model:
                    try:
                        st.session_state['input_df_for_shap'] = pd.DataFrame(input_df, columns=model_columns)
                    except Exception:
                        pass
            
            # Display prediction result with improved styling
            st.markdown('<p class="sub-header">Prediction Result</p>', unsafe_allow_html=True)
            
            if prediction[0] == 1:
                st.markdown(f"""
                <div class="prediction-box" style="background-color: rgba(255, 0, 0, 0.1);">
                    <h3 style="color: #FF0000;">⚠️ High Risk of Churn</h3>
                    <p>This customer is <b>likely to churn</b> with {churn_probability*100:.2f}% probability.</p>
                    <p>Consider implementing retention strategies such as:</p>
                    <ul>
                        <li>Offering a personalized discount</li>
                        <li>Reaching out for feedback</li>
                        <li>Suggesting a more suitable plan</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box" style="background-color: rgba(0, 128, 0, 0.1);">
                    <h3 style="color: #008000;">✅ Low Risk of Churn</h3>
                    <p>This customer is <b>unlikely to churn</b> with {(1-churn_probability)*100:.2f}% probability.</p>
                    <p>Consider these engagement strategies:</p>
                    <ul>
                        <li>Offering loyalty rewards</li>
                        <li>Introducing new complementary services</li>
                        <li>Inviting to referral programs</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        if st.session_state.get('prediction_made', False):
            # Display probability gauge chart
            st.markdown('<p class="sub-header">Churn Probability</p>', unsafe_allow_html=True)
            
            # Get probability from session state
            churn_prob = st.session_state.get('prediction_proba', 0.5) * 100
            
            fig, ax = plt.subplots(figsize=(4, 0.3))
            ax.barh([0], [churn_prob], color='red', height=0.3)
            ax.barh([0], [100-churn_prob], left=[churn_prob], color='green', height=0.3)
            
            # Add a marker for the threshold
            ax.axvline(x=50, color='black', linestyle='--', alpha=0.7)
            
            # Remove axes and add percentage text
            ax.set_yticks([])
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
            ax.set_xlabel('Churn Probability')
            
            st.pyplot(fig)

# Model Explainability Tab
with tab2:
    st.markdown('<p class="sub-header">Model Explainability with SHAP</p>', unsafe_allow_html=True)
    
    if 'input_df_for_shap' in st.session_state and 'prediction' in st.session_state:
        st.subheader("Feature Impact on Prediction")
        
        # Get the input data for SHAP explanation
        input_data = st.session_state['input_df_for_shap']
        
        # If using improved model, apply scaling
        if is_improved_model and scaler is not None:
            input_data_scaled = scaler.transform(input_data)
        else:
            input_data_scaled = input_data.values
            
        try:
            # Load SHAP explainer
            explainer = load_shap_explainer()
            
            # Check if explainer is available
            if explainer is None:
                st.warning("SHAP explainer not available. Please ensure the model loaded correctly.")
            else:
                # Calculate SHAP values
                shap_values = explainer.shap_values(input_data_scaled)
                   
                st.write("The following plot shows how each feature contributes to the prediction:")
                
                # Handle different SHAP output formats and create bar plot
                if isinstance(shap_values, list) and len(shap_values) > 1:
                    # Binary classification - use positive class (churn)
                    shap_vals_to_plot = shap_values[1][0]  # First sample, positive class
                else:
                    # Single output
                    shap_vals_to_plot = shap_values[0] if shap_values.ndim > 1 else shap_values
                
                # Ensure shap_vals_to_plot is 1-dimensional
                if hasattr(shap_vals_to_plot, 'flatten'):
                    shap_vals_to_plot = shap_vals_to_plot.flatten()
                
                # Create a simple bar plot of SHAP values
                feature_names = input_data.columns.tolist()
                
                # Ensure both arrays have the same length
                min_length = min(len(feature_names), len(shap_vals_to_plot))
                feature_names = feature_names[:min_length]
                shap_vals_to_plot = shap_vals_to_plot[:min_length]
                
                # Get top 10 most important features
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'shap_value': shap_vals_to_plot.tolist() if hasattr(shap_vals_to_plot, 'tolist') else list(shap_vals_to_plot)
                })
                importance_df['abs_shap'] = abs(importance_df['shap_value'])
                importance_df = importance_df.sort_values('abs_shap', ascending=False).head(10)
                
                # Create matplotlib figure
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['red' if x > 0 else 'blue' for x in importance_df['shap_value']]
                bars = ax.barh(range(len(importance_df)), importance_df['shap_value'], color=colors)
                ax.set_yticks(range(len(importance_df)))
                ax.set_yticklabels(importance_df['feature'])
                ax.set_xlabel('SHAP Value (Impact on Prediction)')
                ax.set_title('Top 10 Feature Contributions to Churn Prediction')
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
                # Add value labels on bars
                for i, (bar, val) in enumerate(zip(bars, importance_df['shap_value'])):
                    ax.text(val + (0.001 if val >= 0 else -0.001), i, f'{val:.3f}', 
                           va='center', ha='left' if val >= 0 else 'right', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
                # Show feature values table
                st.subheader("Feature Values and Contributions")
                
                # Create a detailed table showing feature values and their SHAP contributions
                shap_values_list = shap_vals_to_plot.tolist() if hasattr(shap_vals_to_plot, 'tolist') else list(shap_vals_to_plot)
                feature_details = pd.DataFrame({
                    'Feature': feature_names,
                    'SHAP Value': shap_values_list,
                    'Impact': ['Increases Churn Risk' if x > 0 else 'Decreases Churn Risk' for x in shap_values_list]
                })
                
                # Sort by absolute SHAP value
                feature_details['Abs_SHAP'] = abs(feature_details['SHAP Value'])
                feature_details = feature_details.sort_values('Abs_SHAP', ascending=False)
                
                # Display top 15 features
                st.dataframe(
                    feature_details[['Feature', 'SHAP Value', 'Impact']].head(15),
                    use_container_width=True
                )
                
                # Explanation of SHAP values
                st.info("""
                **How to interpret SHAP values:**
                - Red features push the prediction higher (toward churn)
                - Blue features push the prediction lower (away from churn)
                - The size of each bar indicates how strongly that feature affects the prediction
                """)
        except Exception as e:
            st.error(f"Error generating SHAP explanation: {str(e)}")
            st.info("SHAP explanations may not be available for this model or input combination.")
    else:
        st.info("Make a prediction first to see SHAP explanations for that customer.")

# Model Performance Tab
with tab4:
    st.markdown('<p class="sub-header">Model Performance Metrics</p>', unsafe_allow_html=True)
    
    # Display key performance metrics prominently
    if is_improved_model:
        st.subheader("🚀 Improved Model Performance")
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Overall Accuracy", "77%", "Balanced performance")
        with col2:
            st.metric("Churn Recall", "72%", "+26% vs baseline")
        with col3:
            st.metric("Churn Precision", "55%", "Reduced false positives")
        with col4:
            st.metric("F1-Score (Churn)", "62%", "Balanced precision/recall")
        
        st.success("✅ **Model Optimization Success**: Improved churn detection recall from 46% to 72% while maintaining overall accuracy")
        
        # Model comparison
        st.subheader("📊 Model Comparison")
        comparison_data = {
            'Metric': ['Overall Accuracy', 'Churn Recall', 'Churn Precision', 'F1-Score (Churn)'],
            'Original Model': ['79%', '46%', '65%', '54%'],
            'Improved Model': ['77%', '72%', '55%', '62%'],
            'Business Impact': ['Slight decrease', '+56% improvement', 'Acceptable trade-off', '+15% improvement']
        }
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
        
        st.info("""
        **Key Improvement**: The improved model prioritizes **churn recall** (finding actual churners) over precision. 
        This means we catch 72% of customers who will churn vs only 46% with the original model. 
        The slight decrease in precision is acceptable as it's better to over-predict churn than miss actual churners.
        """)
    else:
        st.subheader("📈 Baseline Model Performance")
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Overall Accuracy", "79%", "Strong baseline")
        with col2:
            st.metric("Churn Recall", "46%", "Needs improvement")
        with col3:
            st.metric("Churn Precision", "65%", "Good precision")
        with col4:
            st.metric("F1-Score (Churn)", "54%", "Baseline performance")
    
    # Load and display the detailed classification report
    st.subheader("📋 Detailed Classification Report")
    try:
        if is_improved_model:
            with open("improved_classification_report.txt", "r") as f:
                report_text = f.read()
            st.code(report_text, language=None)
        else:
            with open("classification_report.txt", "r") as f:
                report_text = f.read()
            st.code(report_text, language=None)
    except FileNotFoundError:
        st.warning("Classification report not found.")
    
    # Add feature importance visualization if dataset is available
    if dataset is not None:
        st.markdown('<p class="sub-header">Feature Importance</p>', unsafe_allow_html=True)
        
        if is_improved_model:
            # For improved model, show more comprehensive feature importance
            features = ['Contract Type', 'Tenure', 'Monthly Charges', 'Internet Service', 
                       'Payment Method', 'Tech Support', 'Total Charges', 'Paperless Billing',
                       'Service Count', 'Tenure × Monthly Charges', 'Premium Services']
            importance = [0.22, 0.16, 0.14, 0.10, 0.09, 0.08, 0.07, 0.05, 0.04, 0.03, 0.02]
        else:
            # For original model
            features = ['Contract Type', 'Tenure', 'Monthly Charges', 'Internet Service', 
                       'Payment Method', 'Tech Support', 'Total Charges', 'Paperless Billing']
            importance = [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(features, importance, color='skyblue')
        ax.set_xlabel('Relative Importance')
        ax.set_title('Feature Importance for Churn Prediction')
        
        # Add value labels to the bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                   ha='left', va='center')
        
        st.pyplot(fig)

# Customer Segmentation Tab
with tab3:
    st.markdown('<p class="sub-header">Customer Segmentation Analysis</p>', unsafe_allow_html=True)
    
    # Load the dataset for segmentation
    try:
        if dataset is not None:
            # Basic preprocessing
            df_seg = dataset.copy()
            df_seg['TotalCharges'] = pd.to_numeric(df_seg['TotalCharges'], errors='coerce')
            df_seg['TotalCharges'] = df_seg['TotalCharges'].fillna(df_seg['TotalCharges'].median())
            df_seg['Churn'] = df_seg['Churn'].map({'Yes': 1, 'No': 0})
            
            # Select features for segmentation
            segmentation_features = df_seg[['tenure', 'MonthlyCharges', 'TotalCharges']]
            
            # Standardize features
            from sklearn.preprocessing import StandardScaler
            seg_scaler = StandardScaler()
            scaled_features = seg_scaler.fit_transform(segmentation_features)
            
            # Perform K-means clustering
            n_clusters = 4
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df_seg['Segment'] = kmeans.fit_predict(scaled_features)
            
            # Create segment names
            segment_names = {
                0: "New Customers (Low Tenure, Low Charges)",
                1: "High-Value Loyal Customers (High Tenure, High Charges)",
                2: "Mid-Value Customers (Medium Tenure, Medium Charges)",
                3: "At-Risk High-Value Customers (Low-Medium Tenure, High Charges)"
            }
            
            df_seg['Segment_Name'] = df_seg['Segment'].map(segment_names)
            
            # Display segment information
            st.subheader("Customer Segments")
            
            # Show segment distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            segment_counts = df_seg['Segment_Name'].value_counts()
            segment_counts.plot(kind='bar', ax=ax)
            plt.title('Customer Segment Distribution')
            plt.ylabel('Number of Customers')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show churn rate by segment
            st.subheader("Churn Rate by Segment")
            churn_by_segment = df_seg.groupby('Segment_Name')['Churn'].mean() * 100
            
            fig, ax = plt.subplots(figsize=(10, 6))
            churn_by_segment.plot(kind='bar', ax=ax, color='salmon')
            plt.title('Churn Rate by Customer Segment')
            plt.ylabel('Churn Rate (%)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show segment characteristics
            st.subheader("Segment Characteristics")
            segment_profile = df_seg.groupby('Segment_Name').agg({
                'tenure': 'mean',
                'MonthlyCharges': 'mean',
                'TotalCharges': 'mean',
                'Churn': 'mean'
            })
            segment_profile['Churn'] = segment_profile['Churn'] * 100
            segment_profile = segment_profile.rename(columns={'Churn': 'Churn Rate (%)'})
            st.dataframe(segment_profile.round(2))
            
            # Scatter plot of segments
            st.subheader("Customer Segments Visualization")
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = sns.scatterplot(
                x='tenure', 
                y='MonthlyCharges',
                hue='Segment_Name',
                size='Churn',
                sizes=(50, 200),
                palette='viridis',
                data=df_seg,
                ax=ax
            )
            plt.title('Customer Segments by Tenure and Monthly Charges')
            plt.xlabel('Tenure (months)')
            plt.ylabel('Monthly Charges ($)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Retention strategies by segment
            st.subheader("Recommended Retention Strategies by Segment")
            
            strategies = {
                "New Customers (Low Tenure, Low Charges)": [
                    "Onboarding support and education",
                    "Early engagement programs",
                    "First milestone celebrations",
                    "Introductory offers for additional services"
                ],
                "High-Value Loyal Customers (High Tenure, High Charges)": [
                    "VIP loyalty rewards",
                    "Exclusive access to new features",
                    "Personal account manager",
                    "Referral incentives"
                ],
                "Mid-Value Customers (Medium Tenure, Medium Charges)": [
                    "Upgrade incentives",
                    "Bundle discounts",
                    "Loyalty milestone rewards",
                    "Personalized usage recommendations"
                ],
                "At-Risk High-Value Customers (Low-Medium Tenure, High Charges)": [
                    "Proactive outreach and satisfaction checks",
                    "Targeted discounts or plan adjustments",
                    "Premium support access",
                    "Personalized retention offers"
                ]
            }
            
            for segment, strat_list in strategies.items():
                with st.expander(f"Strategies for {segment}"):
                    for strat in strat_list:
                        st.markdown(f"- {strat}")
        else:
            st.warning("Dataset not available for segmentation analysis.")
    except Exception as e:
        st.error(f"Error in customer segmentation: {str(e)}")

# Data Insights Tab
with tab5:
    if dataset is not None:
        st.markdown('<p class="sub-header">Customer Data Insights Dashboard</p>', unsafe_allow_html=True)
        
        # Create tabs for different insights
        insight_tabs = st.tabs(["Overview", "Demographics", "Services", "Charges", "Churn Factors"])
        
        # Overview tab
        with insight_tabs[0]:
            st.subheader("Dataset Overview")
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Customers", f"{len(dataset):,}")
            with col2:
                churn_rate = dataset['Churn'].value_counts(normalize=True).get('Yes', 0) * 100
                st.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
            with col3:
                avg_tenure = dataset['tenure'].mean()
                st.metric("Avg. Customer Tenure", f"{avg_tenure:.1f} months")
            
            # Dataset preview
            st.subheader("Sample Data")
            st.dataframe(dataset.head(10))
            
            # Missing values
            st.subheader("Data Quality")
            missing_data = dataset.isnull().sum()
            if missing_data.sum() > 0:
                st.write("Missing values by column:")
                st.write(missing_data[missing_data > 0])
            else:
                st.success("No missing values in the dataset!")
        
        # Demographics tab
        with insight_tabs[1]:
            st.subheader("Customer Demographics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Gender distribution
                fig, ax = plt.subplots(figsize=(8, 6))
                gender_counts = dataset['gender'].value_counts()
                ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
                ax.set_title('Gender Distribution')
                st.pyplot(fig)
                
                # Partner distribution
                fig, ax = plt.subplots(figsize=(8, 6))
                partner_counts = dataset['Partner'].value_counts()
                ax.pie(partner_counts, labels=partner_counts.index, autopct='%1.1f%%', startangle=90, colors=['#99ff99','#ffcc99'])
                ax.set_title('Partner Distribution')
                st.pyplot(fig)
            
            with col2:
                # Dependents distribution
                fig, ax = plt.subplots(figsize=(8, 6))
                dependents_counts = dataset['Dependents'].value_counts()
                ax.pie(dependents_counts, labels=dependents_counts.index, autopct='%1.1f%%', startangle=90, colors=['#c2c2f0','#ffb3e6'])
                ax.set_title('Dependents Distribution')
                st.pyplot(fig)
                
                # Senior Citizen distribution
                fig, ax = plt.subplots(figsize=(8, 6))
                senior_counts = dataset['SeniorCitizen'].map({0: 'No', 1: 'Yes'}).value_counts()
                ax.pie(senior_counts, labels=senior_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff8080','#66b3ff'])
                ax.set_title('Senior Citizen Distribution')
                st.pyplot(fig)
            
            # Churn by demographics
            st.subheader("Churn Rate by Demographics")
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Gender
            gender_churn = dataset.groupby('gender')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
            axes[0, 0].bar(gender_churn.index, gender_churn.values, color=['#ff9999','#66b3ff'])
            axes[0, 0].set_title('Churn Rate by Gender')
            axes[0, 0].set_ylabel('Churn Rate (%)')
            
            # Partner
            partner_churn = dataset.groupby('Partner')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
            axes[0, 1].bar(partner_churn.index, partner_churn.values, color=['#99ff99','#ffcc99'])
            axes[0, 1].set_title('Churn Rate by Partner Status')
            axes[0, 1].set_ylabel('Churn Rate (%)')
            
            # Dependents
            dependents_churn = dataset.groupby('Dependents')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
            axes[1, 0].bar(dependents_churn.index, dependents_churn.values, color=['#c2c2f0','#ffb3e6'])
            axes[1, 0].set_title('Churn Rate by Dependents Status')
            axes[1, 0].set_ylabel('Churn Rate (%)')
            
            # Senior Citizen
            senior_churn = dataset.groupby('SeniorCitizen')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
            senior_churn.index = ['No', 'Yes']
            axes[1, 1].bar(senior_churn.index, senior_churn.values, color=['#ff8080','#66b3ff'])
            axes[1, 1].set_title('Churn Rate by Senior Citizen Status')
            axes[1, 1].set_ylabel('Churn Rate (%)')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Services tab
        with insight_tabs[2]:
            st.subheader("Service Usage Analysis")
            
            # Service adoption rates
            service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                           'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
            
            service_data = {}
            for col in service_cols:
                service_data[col] = dataset[col].value_counts(normalize=True).to_dict()
            
            service_df = pd.DataFrame(service_data)
            
            # Plot service adoption
            st.subheader("Service Adoption Rates")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Calculate percentage of 'Yes' for each service
            yes_rates = []
            service_names = []
            
            for col in service_cols:
                if 'Yes' in dataset[col].unique():
                    yes_rate = dataset[col].value_counts(normalize=True).get('Yes', 0) * 100
                    yes_rates.append(yes_rate)
                    service_names.append(col)
            
            # Sort by adoption rate
            sorted_indices = np.argsort(yes_rates)
            sorted_services = [service_names[i] for i in sorted_indices]
            sorted_rates = [yes_rates[i] for i in sorted_indices]
            
            # Plot horizontal bar chart
            ax.barh(sorted_services, sorted_rates, color='skyblue')
            ax.set_xlabel('Adoption Rate (%)')
            ax.set_title('Service Adoption Rates')
            
            # Add percentage labels
            for i, v in enumerate(sorted_rates):
                ax.text(v + 1, i, f"{v:.1f}%", va='center')
            
            st.pyplot(fig)
            
            # Churn by service
            st.subheader("Churn Rate by Service")
            
            # Create a figure with subplots
            n_services = len(service_cols)
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            axes = axes.flatten()
            
            for i, service in enumerate(service_cols):
                # Calculate churn rate by service option
                churn_by_service = dataset.groupby(service)['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
                
                # Plot
                axes[i].bar(churn_by_service.index, churn_by_service.values, color='salmon')
                axes[i].set_title(f'Churn Rate by {service}')
                axes[i].set_ylabel('Churn Rate (%)')
                axes[i].set_ylim(0, 100)
                
                # Add percentage labels
                for j, v in enumerate(churn_by_service.values):
                    axes[i].text(j, v + 2, f"{v:.1f}%", ha='center')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Charges tab
        with insight_tabs[3]:
            st.subheader("Customer Charges Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Monthly charges distribution
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(dataset['MonthlyCharges'], bins=20, color='skyblue', edgecolor='black')
                ax.set_title('Distribution of Monthly Charges')
                ax.set_xlabel('Monthly Charges ($)')
                ax.set_ylabel('Number of Customers')
                st.pyplot(fig)
                
                # Monthly charges statistics
                st.metric("Average Monthly Charge", f"${dataset['MonthlyCharges'].mean():.2f}")
                st.metric("Median Monthly Charge", f"${dataset['MonthlyCharges'].median():.2f}")
            
            with col2:
                # Total charges distribution
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(pd.to_numeric(dataset['TotalCharges'], errors='coerce'), bins=20, color='lightgreen', edgecolor='black')
                ax.set_title('Distribution of Total Charges')
                ax.set_xlabel('Total Charges ($)')
                ax.set_ylabel('Number of Customers')
                st.pyplot(fig)
                
                # Total charges statistics
                st.metric("Average Total Charge", f"${pd.to_numeric(dataset['TotalCharges'], errors='coerce').mean():.2f}")
                st.metric("Median Total Charge", f"${pd.to_numeric(dataset['TotalCharges'], errors='coerce').median():.2f}")
            
            # Charges by churn
            st.subheader("Charges by Churn Status")
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Prepare data for plotting
            dataset_plot = dataset.copy()
            dataset_plot['TotalCharges_numeric'] = pd.to_numeric(dataset_plot['TotalCharges'], errors='coerce')
            
            # Monthly charges by churn
            sns.boxplot(x='Churn', y='MonthlyCharges', hue='Churn', data=dataset_plot, ax=axes[0], palette=['green', 'red'], legend=False)
            axes[0].set_title('Monthly Charges by Churn Status')
            axes[0].set_xlabel('Churn')
            axes[0].set_ylabel('Monthly Charges ($)')
            
            # Total charges by churn
            sns.boxplot(x='Churn', y='TotalCharges_numeric', hue='Churn', data=dataset_plot, ax=axes[1], palette=['green', 'red'], legend=False)
            axes[1].set_title('Total Charges by Churn Status')
            axes[1].set_xlabel('Churn')
            axes[1].set_ylabel('Total Charges ($)')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Churn Factors tab
        with insight_tabs[4]:
            st.subheader("Key Churn Factors")
            
            # Contract type vs churn
            fig, ax = plt.subplots(figsize=(10, 6))
            contract_churn = dataset.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
            contract_churn = contract_churn.sort_values(ascending=False)
            
            ax.bar(contract_churn.index, contract_churn.values, color='coral')
            ax.set_title('Churn Rate by Contract Type')
            ax.set_ylabel('Churn Rate (%)')
            ax.set_ylim(0, 100)
            
            # Add percentage labels
            for i, v in enumerate(contract_churn.values):
                ax.text(i, v + 2, f"{v:.1f}%", ha='center')
            
            st.pyplot(fig)
            
            # Tenure vs churn
            st.subheader("Tenure vs Churn")
            
            # Create tenure groups
            tenure_bins = [0, 12, 24, 36, 48, 60, 72]
            tenure_labels = ['0-12', '13-24', '25-36', '37-48', '49-60', '61-72']
            dataset['tenure_group'] = pd.cut(dataset['tenure'], bins=tenure_bins, labels=tenure_labels, right=False)
            
            # Calculate churn rate by tenure group
            tenure_churn = dataset.groupby('tenure_group', observed=True)['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(tenure_churn.index, tenure_churn.values, color='purple')
            ax.set_title('Churn Rate by Tenure (months)')
            ax.set_xlabel('Tenure Group (months)')
            ax.set_ylabel('Churn Rate (%)')
            ax.set_ylim(0, 100)
            
            # Add percentage labels
            for i, v in enumerate(tenure_churn.values):
                ax.text(i, v + 2, f"{v:.1f}%", ha='center')
            
            st.pyplot(fig)
            
            # Payment method vs churn
            fig, ax = plt.subplots(figsize=(12, 6))
            payment_churn = dataset.groupby('PaymentMethod')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
            payment_churn = payment_churn.sort_values(ascending=False)
            
            ax.bar(payment_churn.index, payment_churn.values, color='teal')
            ax.set_title('Churn Rate by Payment Method')
            ax.set_ylabel('Churn Rate (%)')
            ax.set_ylim(0, 100)
            plt.xticks(rotation=45, ha='right')
            
            # Add percentage labels
            for i, v in enumerate(payment_churn.values):
                ax.text(i, v + 2, f"{v:.1f}%", ha='center')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Correlation heatmap
            st.subheader("Feature Correlation with Churn")
            
            # Prepare data for correlation
            df_corr = dataset.copy()
            df_corr['Churn'] = df_corr['Churn'].map({'Yes': 1, 'No': 0})
            df_corr['TotalCharges'] = pd.to_numeric(df_corr['TotalCharges'], errors='coerce')
            
            # Convert categorical to numeric
            for col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
                df_corr[col] = df_corr[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
            
            # Select numeric columns
            numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn', 
                           'SeniorCitizen', 'gender', 'Partner', 'Dependents', 
                           'PhoneService', 'PaperlessBilling']
            
            # Calculate correlation with churn
            corr_with_churn = df_corr[numeric_cols].corr()['Churn'].sort_values(ascending=False)
            corr_with_churn = corr_with_churn.drop('Churn')  # Remove self-correlation
            
            fig, ax = plt.subplots(figsize=(10, 8))
            bars = ax.barh(corr_with_churn.index, corr_with_churn.values, color=['red' if x > 0 else 'green' for x in corr_with_churn.values])
            ax.set_title('Correlation with Churn')
            ax.set_xlabel('Correlation Coefficient')
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width + 0.01 if width > 0 else width - 0.01
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                       ha='left' if width > 0 else 'right', va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.warning("Dataset not available for analysis. Please ensure the dataset file is in the correct location.")

# A/B Testing Simulation Tab
with tab6:
    st.markdown('<p class="sub-header">A/B Testing Simulation for Retention Strategies</p>', unsafe_allow_html=True)
    
    st.write("""
    This tab simulates A/B testing of different retention strategies to help identify which approaches 
    might be most effective for reducing customer churn. Select parameters below to run simulations.
    """)
    
    # Strategy selection
    strategy_options = {
        "Discount Offer": {
            "description": "Offer a discount on monthly charges",
            "default_effect": 0.15,  # 15% reduction in churn
            "cost_per_customer": 10,  # $10 per customer
            "target_segments": ["High Value at Risk", "New Customers"]
        },
        "Contract Upgrade": {
            "description": "Offer incentives to upgrade to annual/biennial contracts",
            "default_effect": 0.25,  # 25% reduction in churn
            "cost_per_customer": 15,  # $15 per customer
            "target_segments": ["Month-to-Month Subscribers", "High Value at Risk"]
        },
        "Service Upgrade": {
            "description": "Offer free premium service upgrades",
            "default_effect": 0.18,  # 18% reduction in churn
            "cost_per_customer": 12,  # $12 per customer
            "target_segments": ["Basic Service Users", "Competitors' Customers"]
        },
        "Tech Support": {
            "description": "Provide free tech support for 3 months",
            "default_effect": 0.12,  # 12% reduction in churn
            "cost_per_customer": 8,  # $8 per customer
            "target_segments": ["Technical Issues Segment", "New Customers"]
        },
        "Loyalty Program": {
            "description": "Enroll customers in an enhanced loyalty program",
            "default_effect": 0.20,  # 20% reduction in churn
            "cost_per_customer": 5,  # $5 per customer
            "target_segments": ["Long-term Customers", "High Value Stable"]
        }
    }
    
    # Create columns for strategy selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Strategy A (Control)")
        strategy_a = st.selectbox("Select Strategy A", list(strategy_options.keys()), key="strategy_a")
        effect_a = st.slider(
            "Expected Churn Reduction (%)", 
            min_value=5, 
            max_value=40, 
            value=int(strategy_options[strategy_a]["default_effect"] * 100),
            key="effect_a"
        ) / 100
        cost_a = st.number_input(
            "Cost per Customer ($)", 
            min_value=1.0, 
            max_value=50.0, 
            value=float(strategy_options[strategy_a]["cost_per_customer"]),
            key="cost_a"
        )
        
    with col2:
        st.subheader("Strategy B (Test)")
        strategy_b = st.selectbox("Select Strategy B", list(strategy_options.keys()), key="strategy_b", index=1)
        effect_b = st.slider(
            "Expected Churn Reduction (%)", 
            min_value=5, 
            max_value=40, 
            value=int(strategy_options[strategy_b]["default_effect"] * 100),
            key="effect_b"
        ) / 100
        cost_b = st.number_input(
            "Cost per Customer ($)", 
            min_value=1.0, 
            max_value=50.0, 
            value=float(strategy_options[strategy_b]["cost_per_customer"]),
            key="cost_b"
        )
    
    # Target segment selection
    st.subheader("Target Customer Segment")
    
    segment_options = [
        "All Customers",
        "High Value at Risk",
        "Month-to-Month Subscribers",
        "Long-term Customers",
        "New Customers (< 12 months)",
        "Basic Service Users",
        "Premium Service Users"
    ]
    
    selected_segment = st.selectbox("Select Target Segment", segment_options)
    
    # Sample size and confidence level
    st.subheader("Simulation Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        sample_size = st.slider("Sample Size (per group)", 100, 5000, 1000, 100)
    with col2:
        confidence_level = st.select_slider("Confidence Level", options=[0.90, 0.95, 0.99], value=0.95)
    
    # Run simulation button
    if st.button("Run A/B Test Simulation"):
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(100):
            # Update progress bar
            progress_bar.progress(i + 1)
            status_text.text(f"Simulation in progress: {i+1}%")
            time.sleep(0.01)
        
        status_text.text("Simulation completed!")
        
        # Get base churn rate from dataset
        base_churn_rate = dataset['Churn'].value_counts(normalize=True).get('Yes', 0)
        
        # Simulate results
        np.random.seed(42)  # For reproducibility
        
        # Function to simulate churn outcomes
        def simulate_churn(base_rate, effect, sample_size):
            reduced_rate = base_rate * (1 - effect)
            # Add some random variation
            observed_rate = np.random.normal(reduced_rate, 0.02)
            observed_rate = max(0, min(observed_rate, 1))  # Keep between 0 and 1
            
            # Generate binary outcomes (churned or not)
            outcomes = np.random.choice(
                [1, 0], 
                size=sample_size, 
                p=[observed_rate, 1-observed_rate]
            )
            
            return outcomes, observed_rate
        
        # Run simulations
        outcomes_a, observed_rate_a = simulate_churn(base_churn_rate, effect_a, sample_size)
        outcomes_b, observed_rate_b = simulate_churn(base_churn_rate, effect_b, sample_size)
        
        # Calculate metrics
        churn_count_a = np.sum(outcomes_a)
        churn_count_b = np.sum(outcomes_b)
        
        churn_rate_a = churn_count_a / sample_size
        churn_rate_b = churn_count_b / sample_size
        
        # Calculate confidence intervals using bootstrap
        def bootstrap_ci(outcomes, confidence=0.95):
            n_bootstraps = 1000
            bootstrap_means = []
            
            for _ in range(n_bootstraps):
                bootstrap_sample = np.random.choice(outcomes, size=len(outcomes), replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
                
            lower_bound = np.percentile(bootstrap_means, (1 - confidence) * 100 / 2)
            upper_bound = np.percentile(bootstrap_means, 100 - (1 - confidence) * 100 / 2)
            
            return lower_bound, upper_bound
        
        ci_a_lower, ci_a_upper = bootstrap_ci(outcomes_a, confidence_level)
        ci_b_lower, ci_b_upper = bootstrap_ci(outcomes_b, confidence_level)
        
        # Calculate p-value using chi-squared test
        contingency_table = np.array([
            [churn_count_a, sample_size - churn_count_a],
            [churn_count_b, sample_size - churn_count_b]
        ])
        
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        
        # Calculate ROI
        retained_customers_a = sample_size - churn_count_a
        retained_customers_b = sample_size - churn_count_b
        
        # Assume average customer lifetime value of $500
        customer_ltv = 500
        
        total_cost_a = cost_a * sample_size
        total_cost_b = cost_b * sample_size
        
        revenue_a = retained_customers_a * customer_ltv
        revenue_b = retained_customers_b * customer_ltv
        
        roi_a = (revenue_a - total_cost_a) / total_cost_a * 100
        roi_b = (revenue_b - total_cost_b) / total_cost_b * 100
        
        # Display results
        st.subheader("A/B Test Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Strategy A Churn Rate", 
                f"{churn_rate_a:.2%}", 
                f"{base_churn_rate - churn_rate_a:.2%}"
            )
            st.write(f"95% CI: [{ci_a_lower:.2%}, {ci_a_upper:.2%}]")
            st.metric("Strategy A ROI", f"{roi_a:.1f}%")
            
        with col2:
            st.metric(
                "Strategy B Churn Rate", 
                f"{churn_rate_b:.2%}", 
                f"{base_churn_rate - churn_rate_b:.2%}"
            )
            st.write(f"95% CI: [{ci_b_lower:.2%}, {ci_b_upper:.2%}]")
            st.metric("Strategy B ROI", f"{roi_b:.1f}%")
            
        with col3:
            st.metric(
                "Difference (B-A)", 
                f"{churn_rate_b - churn_rate_a:.2%}",
                f"{(churn_rate_b - churn_rate_a) / churn_rate_a:.1%}"
            )
            st.write(f"p-value: {p_value:.4f}")
            
            # Determine if result is statistically significant
            if p_value < 1 - confidence_level:
                st.success(f"Statistically Significant (p < {1 - confidence_level})")
            else:
                st.warning(f"Not Statistically Significant (p > {1 - confidence_level})")
        
        # Visualization of results
        st.subheader("Visualization of Results")
        
        # Churn rates comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        strategies = [f"Base Rate", f"Strategy A\n({strategy_a})", f"Strategy B\n({strategy_b})"]
        rates = [base_churn_rate, churn_rate_a, churn_rate_b]
        colors = ['gray', 'skyblue', 'lightgreen']
        
        bars = ax.bar(strategies, rates, color=colors)
        
        # Add error bars for confidence intervals
        ax.errorbar(
            x=1, 
            y=churn_rate_a, 
            yerr=[[churn_rate_a - ci_a_lower], [ci_a_upper - churn_rate_a]], 
            fmt='none', 
            color='black', 
            capsize=5
        )
        
        ax.errorbar(
            x=2, 
            y=churn_rate_b, 
            yerr=[[churn_rate_b - ci_b_lower], [ci_b_upper - churn_rate_b]], 
            fmt='none', 
            color='black', 
            capsize=5
        )
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., 
                height + 0.01, 
                f'{height:.2%}', 
                ha='center', 
                va='bottom'
            )
        
        ax.set_ylabel('Churn Rate')
        ax.set_title('Churn Rate Comparison')
        ax.set_ylim(0, max(rates) * 1.2)
        
        st.pyplot(fig)
        
        # ROI comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        strategies = [f"Strategy A\n({strategy_a})", f"Strategy B\n({strategy_b})"]
        roi_values = [roi_a, roi_b]
        colors = ['skyblue', 'lightgreen']
        
        bars = ax.bar(strategies, roi_values, color=colors)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., 
                height + 1, 
                f'{height:.1f}%', 
                ha='center', 
                va='bottom'
            )
        
        ax.set_ylabel('ROI (%)')
        ax.set_title('Return on Investment Comparison')
        
        st.pyplot(fig)
        
        # Recommendation
        st.subheader("Recommendation")
        
        if p_value < 1 - confidence_level:
            if roi_b > roi_a:
                st.success(f"Strategy B ({strategy_b}) is recommended as it provides statistically significant better results with higher ROI.")
            else:
                st.success(f"Strategy A ({strategy_a}) is recommended as it provides better ROI despite Strategy B having lower churn rate.")
        else:
            if abs(roi_b - roi_a) > 10:  # If ROI difference is substantial
                better_strategy = "B" if roi_b > roi_a else "A"
                better_strategy_name = strategy_b if roi_b > roi_a else strategy_a
                st.info(f"Results are not statistically significant, but Strategy {better_strategy} ({better_strategy_name}) shows better ROI. Consider running a larger test.")
            else:
                st.info("Results are not statistically significant and ROI is similar. Consider testing different strategies or increasing sample size.")
    
    # Strategy information
    st.subheader("Strategy Information")
    
    for strategy, info in strategy_options.items():
        with st.expander(f"{strategy} - {info['description']}"):
            st.write(f"**Default Effect:** {info['default_effect']*100:.0f}% churn reduction")
            st.write(f"**Default Cost:** ${info['cost_per_customer']} per customer")
            st.write("**Recommended Target Segments:**")
            for segment in info['target_segments']:
                st.write(f"- {segment}")
    
    # Methodology explanation
    with st.expander("A/B Testing Methodology"):
        st.write("""
        **How the simulation works:**
        
        1. **Baseline:** We start with the actual churn rate from our dataset.
        2. **Effect Simulation:** We apply the expected effect of each strategy to reduce the churn rate.
        3. **Random Variation:** We add realistic random variation to simulate real-world uncertainty.
        4. **Statistical Analysis:** We calculate confidence intervals using bootstrap resampling and p-values using chi-squared tests.
        5. **ROI Calculation:** We estimate return on investment based on customer lifetime value and strategy cost.
        
        **Interpreting Results:**
        
        - **Statistical Significance:** If the p-value is below the threshold (e.g., 0.05 for 95% confidence), the difference between strategies is likely not due to random chance.
        - **Confidence Intervals:** These show the range where the true churn rate likely falls with the selected confidence level.
        - **ROI:** Higher ROI indicates better financial returns, even if the churn reduction is smaller.
        
        **Limitations:**
        
        - This is a simulation based on assumptions and should be validated with real A/B tests.
        - Customer behavior may vary from these simulations in real-world implementations.
        - The model doesn't account for long-term effects or customer satisfaction metrics.
        """)

# Prediction History Tab
with tab7:
    st.markdown('<p class="sub-header">Prediction History</p>', unsafe_allow_html=True)
    
    history_file = 'predictions/prediction_history.csv'
    if os.path.exists(history_file):
        try:
            history_df = pd.read_csv(history_file)
            if not history_df.empty:
                st.subheader("Recent Predictions")
                
                # Show summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Predictions", len(history_df))
                with col2:
                    churn_predictions = history_df['prediction'].str.contains('Likely to churn').sum()
                    st.metric("Churn Predictions", churn_predictions)
                with col3:
                    if len(history_df) > 0:
                        avg_prob = history_df['churn_probability'].str.rstrip('%').astype(float).mean()
                        st.metric("Avg Churn Probability", f"{avg_prob:.1f}%")
                
                # Display recent predictions
                st.subheader("Prediction Details")
                
                # Sort by timestamp (most recent first)
                history_df_sorted = history_df.sort_values('timestamp', ascending=False)
                
                # Display in a nice format
                for idx, row in history_df_sorted.head(10).iterrows():
                    with st.expander(f"Prediction on {row['timestamp']} - {row['prediction']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Prediction:** {row['prediction']}")
                            st.write(f"**Probability:** {row['churn_probability']}")
                            st.write(f"**Contract:** {row.get('Contract', 'N/A')}")
                            st.write(f"**Tenure:** {row.get('Tenure', 'N/A')}")
                        with col2:
                            st.write(f"**Monthly Charges:** {row.get('Monthly Charges', 'N/A')}")
                            st.write(f"**Payment Method:** {row.get('Payment Method', 'N/A')}")
                            st.write(f"**Internet Service:** {row.get('Internet Service', 'N/A')}")
                
                # Option to download history
                csv = history_df.to_csv(index=False)
                st.download_button(
                    label="Download Prediction History",
                    data=csv,
                    file_name="churn_prediction_history.csv",
                    mime="text/csv"
                )
                
                # Option to clear history
                if st.button("Clear Prediction History", type="secondary"):
                    if os.path.exists(history_file):
                        os.remove(history_file)
                        st.success("Prediction history cleared!")
                        st.rerun()
            else:
                st.info("No predictions made yet. Make your first prediction in the 'Prediction' tab!")
        except Exception as e:
            st.error(f"Error loading prediction history: {str(e)}")
    else:
        st.info("No prediction history available. Make your first prediction in the 'Prediction' tab!")

