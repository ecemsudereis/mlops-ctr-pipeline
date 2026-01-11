import unittest
import pandas as pd
import numpy as np
from sklearn.feature_extraction import FeatureHasher
import os
import sys

# Ensure we can import from the same directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from train_pipeline import mock_preprocessing
except ImportError:
    # Fallback if running from a different directory structure
    from CTR_Project.train_pipeline import mock_preprocessing

class TestCTRPipeline(unittest.TestCase):
    
    # 1. UNIT TEST: Verify Feature Hashing Logic
    def test_feature_hashing(self):
        # Create dummy sample data
        sample_data = pd.DataFrame({'device_id': ['dev1', 'dev2', 'dev1']})
        
        # Apply Feature Hashing
        hasher = FeatureHasher(n_features=10, input_type='string')
        hashed_output = hasher.transform(sample_data['device_id'].apply(lambda x: [x])).toarray()
        
        # Check 1: Is the output shape correct?
        self.assertEqual(hashed_output.shape, (3, 10))
        
        # Check 2: Deterministic behavior
        np.testing.assert_array_equal(hashed_output[0], hashed_output[2])
        
        print("\n✅ Unit Test Passed: Feature Hashing logic is consistent.")

    # 2. SMOKE TEST: Verify Data Pipeline Integrity
    def test_data_pipeline(self):
        try:
            X, y = mock_preprocessing('dummy_path')
            
            self.assertFalse(X.empty, "Feature dataframe (X) should not be empty.")
            self.assertFalse(y.empty, "Target series (y) should not be empty.")
            self.assertEqual(len(X), len(y), "Features and Target must have the same length.")
            
            print("\n✅ Smoke Test Passed: Data pipeline generates valid input/output pairs.")
            
        except Exception as e:
            self.fail(f"Smoke Test Failed with error: {str(e)}")

    # 3. SABOTAGE TEST (For Evidence B)
    def test_sabotage(self):
        # This test will fail intentionally because 1 is not equal to 0.
        print("\n🧨 SABOTAGE ACTIVATED: Deployment will be stopped!")
        self.assertEqual(1, 0, "SABOTAGE: Deployment stopped due to intentional failure! 🛑")

if __name__ == '__main__':
    unittest.main()