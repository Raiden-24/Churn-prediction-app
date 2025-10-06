import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Drop customerID as it's not relevant for prediction
df = df.drop('customerID', axis=1)

# Convert TotalCharges to numeric and handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Feature Engineering
print("Performing feature engineering...")

# 1. Create tenure-related features
df['tenure_group'] = pd.qcut(df['tenure'], 4, labels=['0-1 year', '1-3 years', '3-5 years', '5+ years'])

# 2. Create average monthly charge
df['avg_monthly_charges'] = df['TotalCharges'] / (df['tenure'] + 0.1)  # Adding 0.1 to avoid division by zero

# 3. Create charge difference (how much more/less than average)
df['charge_diff'] = df['MonthlyCharges'] - df['avg_monthly_charges']

# 4. Create service count feature
service_columns = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

# Function to count services
def count_services(row):
    count = 0
    for col in service_columns:
        if row[col] not in ['No', 'No phone service', 'No internet service', 'DSL']:
            count += 1
    return count

df['service_count'] = df.apply(count_services, axis=1)

# 5. Create binary flags for premium services
df['has_premium_tech'] = df['TechSupport'].apply(lambda x: 1 if x == 'Yes' else 0)
df['has_premium_security'] = df['OnlineSecurity'].apply(lambda x: 1 if x == 'Yes' else 0)
df['has_premium_support'] = df[['OnlineBackup', 'DeviceProtection']].apply(
    lambda x: 1 if 'Yes' in x.values else 0, axis=1
)
df['has_streaming'] = df[['StreamingTV', 'StreamingMovies']].apply(
    lambda x: 1 if 'Yes' in x.values else 0, axis=1
)

# 6. Create interaction features
df['tenure_x_monthly'] = df['tenure'] * df['MonthlyCharges']

# Convert categorical variables to numeric
print("Encoding categorical variables...")

# Binary encoding for Yes/No columns
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# One-hot encoding for categorical columns
cat_cols = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaymentMethod', 'tenure_group']

df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Prepare features and target
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Split the data
print("Splitting data and scaling features...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to handle class imbalance
print("Applying SMOTE to handle class imbalance...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"Original training set shape: {np.bincount(y_train)}")
print(f"Resampled training set shape: {np.bincount(y_train_smote)}")

# Train the model
print("Training Random Forest model...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)
rf.fit(X_train_smote, y_train_smote)

# Evaluate the model
print("Evaluating model...")
y_pred = rf.predict(X_test_scaled)
report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)

# Save the model and related files
print("Saving model and related files...")
with open('improved_classification_report.txt', 'w') as f:
    f.write(report)

joblib.dump(rf, 'improved_churn_model.pkl')
joblib.dump(X.columns.tolist(), 'improved_churn_model_columns.pkl')
joblib.dump(scaler, 'improved_churn_model_scaler.pkl')

print("Model training completed successfully!")