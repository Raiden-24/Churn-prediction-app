import unittest
import os
import joblib
import numpy as np
import pandas as pd


class TestArtifactsAndPrediction(unittest.TestCase):
    """Lightweight smoke tests aligned with current artifact names and formats."""

    def test_improved_artifacts_exist_or_fallback(self):
        improved = all([
            os.path.exists('improved_churn_model.pkl'),
            os.path.exists('improved_churn_model_columns.pkl'),
            os.path.exists('improved_churn_model_scaler.pkl'),
        ])
        original = all([
            os.path.exists('churn_model.pkl'),
            os.path.exists('churn_model_columns.pkl'),
        ])
        self.assertTrue(improved or original, "No model artifacts found (neither improved nor original).")

    def test_can_predict_with_available_model(self):
        if os.path.exists('improved_churn_model.pkl'):
            model = joblib.load('improved_churn_model.pkl')
            feature_cols = joblib.load('improved_churn_model_columns.pkl')
            scaler = joblib.load('improved_churn_model_scaler.pkl')

            # Build a minimal feature row (zeros) with correct columns
            X_df = pd.DataFrame([np.zeros(len(feature_cols))], columns=feature_cols)
            X = scaler.transform(X_df)
        elif os.path.exists('churn_model.pkl'):
            model = joblib.load('churn_model.pkl')
            feature_cols = joblib.load('churn_model_columns.pkl')
            X_df = pd.DataFrame([np.zeros(len(feature_cols))], columns=feature_cols)
            X = X_df.values
        else:
            self.skipTest("No model available for prediction test")

        # Model should produce a valid class and probability distribution
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        self.assertIn(int(y_pred[0]), [0, 1])
        self.assertAlmostEqual(float(y_proba[0].sum()), 1.0, places=5)


if __name__ == '__main__':
    unittest.main()