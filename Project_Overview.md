# Project Overview: AI-Powered Customer Churn Prediction Platform

Links: [GitHub Repository](https://github.com/Raiden-24/Churn-prediction-app) • [Live App](https://churn-prediction-app-by-amruth.streamlit.app)

## 1) Executive Summary
- Purpose: Predict which customers are likely to churn so teams can proactively retain them.
- Outcome: 72% churn detection rate (recall), 77% overall accuracy; transparent explanations via SHAP; interactive 7-tab dashboard; production deployment on Streamlit Cloud.
- Business impact (illustrative): For a 10k-customer base, +702 churners saved annually vs baseline (~$700K revenue protection at $1,000 CLV).

## 2) Dataset
- Source: `WA_Fn-UseC_-Telco-Customer-Churn.csv` (Telco customer churn dataset).
- Target: `Churn` (Yes/No → mapped to 1/0 during training).
- Key raw features
  - Numeric: `tenure`, `MonthlyCharges`, `TotalCharges` (converted to numeric, missing handled with median).
  - Binary: `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`.
  - Multi-class categorical: `gender`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaymentMethod`.

## 3) ML Pipeline (Training) – `improved_model.py`
- Cleaning
  - Drop `customerID`.
  - `TotalCharges` → `pd.to_numeric(..., errors='coerce')`, impute median.
- Feature Engineering
  - `tenure_group`: 4 quantile bins via `pd.qcut` with labels: `['0-1 year','1-3 years','3-5 years','5+ years']`.
  - `avg_monthly_charges` = `TotalCharges` / (`tenure` + 0.1).
  - `charge_diff` = `MonthlyCharges` − `avg_monthly_charges`.
  - `service_count`: count of service features that are enabled (ignores values like `No`, `No internet service`, `DSL`).
  - Premium flags: `has_premium_tech`, `has_premium_security`, `has_premium_support`, `has_streaming`.
  - Interaction: `tenure_x_monthly`.
- Encoding
  - Binary map: `['Partner','Dependents','PhoneService','PaperlessBilling','Churn']` to {Yes:1, No:0}.
  - One-hot encode: `gender`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaymentMethod`, `tenure_group` (with `drop_first=True`).
- Split, Scale, Balance
  - `train_test_split(..., stratify=y, test_size=0.2, random_state=42)`.
  - `StandardScaler` on features.
  - `SMOTE(random_state=42)` for class imbalance (approx 73:27 non-churn:churn).
- Model
  - `RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, class_weight='balanced', random_state=42)`.
- Evaluation → Artifacts
  - `classification_report` saved to `improved_classification_report.txt`.
  - Metrics (from artifact): Accuracy 0.77; Churn class: Precision 0.55, Recall 0.72, F1 0.62.
  - Saved: `improved_churn_model.pkl`, `improved_churn_model_columns.pkl`, `improved_churn_model_scaler.pkl`.

## 4) Baseline Model (Reference)
- Artifacts: `churn_model.pkl`, `churn_model_columns.pkl`, `classification_report.txt`.
- Key metrics (from artifact): Accuracy 0.79; Churn recall 0.46; F1 0.54.
- Strategy: Improved model prioritizes recall (detecting churners) with a small accuracy trade-off.

## 5) Application Architecture – `app.py`
- Caching & Loading
  - `load_model()` [`@st.cache_resource`]: Loads improved artifacts if present; else falls back to baseline.
  - `load_shap_explainer()` [`@st.cache_resource`]: Creates `shap.TreeExplainer(model)` separately for reliability.
  - `load_dataset()` [`@st.cache_data`]: Reads CSV for insights tabs.
- Front Page Highlights
  - Prominent metric cards: Accuracy 77%, Churn Recall 72% (+56% vs baseline), Precision 55%, F1 62%.
- Tabs
  - `tab1` Prediction
  - `tab2` Model Explainability (SHAP)
  - `tab3` Customer Segmentation (K-means)
  - `tab4` Model Performance
  - `tab5` Data Insights
  - `tab6` A/B Testing Simulation
  - `tab7` Prediction History

### 5.1 Prediction (`tab1`)
- Inputs: Collected in sidebar (Demographics, Account Info, Services, Charges).
- Preprocessing (improved model path)
  - Recreates engineering in-app: `tenure_group` (manual bands), `avg_monthly_charges`, `charge_diff`, `service_count`, premium flags, `tenure_x_monthly`.
  - Binary map + one-hot encode; ensures alignment with `model_columns` (adds missing zero columns).
  - Scales with persisted `StandardScaler`.
  - Saves unscaled engineered DataFrame to `st.session_state['input_df_for_shap']` for explainability.
- Prediction
  - `model.predict(...)` and `model.predict_proba(...)`.
  - UI: Styled result card (High/Low Risk), probability bar, actionable suggestions.
- Persistence
  - `save_prediction(...)` writes to `predictions/prediction_history.csv` with timestamp, summary, class, probability.

### 5.2 Explainability (`tab2`)
- Uses `st.session_state['input_df_for_shap']`.
- Scales if improved model; else uses raw values.
- SHAP values
  - `explainer.shap_values(input_data_scaled)` with robust handling for list/array outputs.
  - Displays: Top-10 feature contributions horizontal bar chart; contributions table (Increases/Decreases churn risk).

### 5.3 Customer Segmentation (`tab3`)
- Dataset copy → clean: numeric `TotalCharges`, impute; `Churn` → 1/0.
- Features: `tenure`, `MonthlyCharges`, `TotalCharges` → `StandardScaler`.
- `KMeans(n_clusters=4, random_state=42)` → labels mapped to descriptive names.
- Visuals: Segment distribution, churn by segment, segment profile table.

### 5.4 Model Performance (`tab4`)
- Metric cards (contextualized) and a comparison table: Original vs Improved, with business impact notes.
- Full `classification_report` display (`improved_classification_report.txt` or `classification_report.txt`).
- Feature importance (illustrative bar chart) for stakeholder communication.

### 5.5 Data Insights (`tab5`)
- Distributions of key variables; churn factor visualizations (e.g., charges by churn, demographics, services).
- Helps stakeholders understand the data landscape and drivers.

### 5.6 A/B Testing Simulation (`tab6`)
- Simulates retention strategies and measures effect using statistical tests (e.g., chi-squared with `scipy.stats.chi2_contingency`).
- Supports data-driven decisions before costly rollouts.

### 5.7 Prediction History (`tab7`)
- Reads `predictions/prediction_history.csv`.
- Shows summary stats, detailed table, download option; supports auditability and analysis.

## 6) Key Functions (I/O Summary)
- `load_model()` → `(model, model_columns, scaler|None, is_improved_model: bool)`
- `load_shap_explainer()` → `TreeExplainer|None`
- `user_input_features()` → `(X_for_model, user_summary_dict)`
- `save_prediction(user_summary, prediction, prob)` → CSV append to `predictions/prediction_history.csv`

## 7) Metrics & Rationale
- Improved model (artifact): Accuracy 77%, Precision 55%, Recall 72%, F1 62% (churn class).
- Baseline (artifact): Accuracy 79%, Recall 46%.
- Rationale: In churn prevention, **recall** is business-critical (missing a churner is costlier than a false alarm). Hence, we optimize for recall with a small accuracy trade-off.

## 8) Testing – `test_app.py`
- Verifies artifacts exist (improved or fallback original).
- Ensures model can make predictions and outputs valid probability distribution.

## 9) Technology Stack
- Python, Streamlit UI, scikit-learn, pandas/numpy, SHAP, imbalanced-learn, matplotlib/seaborn.
- `requirements.txt` pinned to Streamlit Cloud–compatible versions (Python 3.12 recommended on Cloud).

## 10) Local Setup & Commands
```bash
# Install deps
pip install -r requirements.txt

# Run app
streamlit run app.py

# Run tests
python -m unittest -v
```

## 11) Deployment Notes (Streamlit Cloud)
- Keep `requirements.txt` minimal; avoid building SciPy from source on Python 3.13+ (use Python 3.12 on Cloud).
- Artifacts (`*.pkl` and `*_columns.pkl`, scaler) must be present in repo or storage.
- Streamlit caching (`@st.cache_resource`, `@st.cache_data`) accelerates cold starts.

## 12) Design Choices & Trade-offs
- Prioritized recall over accuracy to maximize saved customers.
- Kept model tree-based (Random Forest) for stability and SHAP compatibility.
- Repeated feature engineering logic in-app to ensure inference-time parity (with alignment to saved `model_columns`).
- Added robust fallbacks and user messaging to reduce runtime failures.

## 13) Future Enhancements
- Threshold tuning control in UI; per-segment thresholds.
- Model registry & versioning (e.g., MLflow) with A/B model testing.
- Data drift monitoring and automated retraining pipelines.
- Persist history to a database, add auth/roles, and export APIs.

## 14) Demo Script (Interview)
1. Go to Prediction → enter a realistic profile → Predict.
2. Show probability bar and tailored suggestions.
3. Go to Explainability → top contributing features (SHAP) for that customer.
4. Model Performance → emphasize 72% recall vs 46% baseline and business impact.
5. Customer Segmentation → which segments are riskiest and how to act.
6. A/B Testing → simulate discount/offer effectiveness (statistical validation).
7. Prediction History → data trail for operations/analytics.

## 15) License
This project is licensed under the MIT License (see `LICENSE`).
