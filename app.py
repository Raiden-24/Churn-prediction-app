import streamlit as st
import pandas as pd
import joblib

# Load the trained model and the column list
try:
    model = joblib.load('churn_model.pkl')
    model_columns = joblib.load('churn_model_columns.pkl')
except FileNotFoundError:
    st.error("Model files not found! Please run the model_training.ipynb notebook first.")
    st.stop()

# Set the page title and icon
st.set_page_config(page_title="Churn Predictor", page_icon="🚀")

# App title and description
st.title('AI-Powered Customer Churn Predictor')
st.markdown("""
This application predicts whether a customer is likely to churn based on their account details.
Adjust the sliders and dropdowns on the left to match a customer's profile and click 'Predict' to see the result.
""")

st.sidebar.header('Model Performance')

# Load and display the classification report
try:
    with open("classification_report.txt", "r") as f:
        report_text = f.read()
    st.sidebar.text("Classification Report:")
    # Use st.code to display the text in a formatted block
    st.sidebar.code(report_text, language=None)
except FileNotFoundError:
    st.sidebar.warning("Classification report not found.")

# --- Sidebar for User Input ---
st.sidebar.header('Customer Account Details')

def user_input_features():
    # Input fields in the sidebar
    tenure = st.sidebar.slider('Tenure (months)', 0, 72, 24)
    monthly_charges = st.sidebar.slider('Monthly Charges ($)', 18.0, 120.0, 70.0, 0.05)
    total_charges = st.sidebar.slider('Total Charges ($)', 18.0, 9000.0, 1400.0, 1.0)
    
    contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    payment_method = st.sidebar.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    tech_support = st.sidebar.selectbox('Tech Support', ('No', 'Yes', 'No internet service'))
    internet_service = st.sidebar.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
    
    # Create a dictionary for the input. We'll start with just a few for simplicity.
    # This dictionary needs to eventually match the columns the model was trained on.
    data = {
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'gender': [0], # Assuming Female as default for simplicity
        'Partner': [0],
        'Dependents': [0],
        'PhoneService': [1],
        'PaperlessBilling': [1],
        # One-hot encoded columns start here. We set the selected one to 1 and others to 0.
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
    
    return features

input_df = user_input_features()

# Display user input in a collapsible section
with st.expander("Show User Input Summary"):
    st.write(input_df)

# Prediction button
if st.button('Predict Churn', key='predict_button'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader('Prediction Result')
    
    if prediction[0] == 1:
        st.error('This customer is **likely to churn**.')
    else:
        st.success('This customer is **unlikely to churn**.')

    st.subheader('Prediction Probability')
    prob_df = pd.DataFrame({
        'Probability': [f"{prediction_proba[0][0]*100:.2f}%", f"{prediction_proba[0][1]*100:.2f}%"]
    }, index=['Not Churning', 'Churning'])
    st.dataframe(prob_df)

