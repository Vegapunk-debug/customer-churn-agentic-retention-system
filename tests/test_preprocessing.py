import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import clean_data, encode_categorical

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.sample_data = pd.DataFrame({
            'customerID': ['123', '456'],
            'gender': ['Female', 'Male'],
            'Partner': ['Yes', 'No'],
            'Dependents': ['No', 'Yes'],
            'PhoneService': ['Yes', 'Yes'],
            'PaperlessBilling': ['Yes', 'No'],
            'TotalCharges': ['100.5', ' '],  # Check empty string handling
            'MultipleLines': ['No', 'Yes'],
            'Churn': ['No', 'Yes']
        })

    def test_clean_data_drops_columns(self):
        cleaned = clean_data(self.sample_data)
        self.assertNotIn('customerID', cleaned.columns)
        self.assertNotIn('Churn', cleaned.columns)

    def test_clean_data_binary_encoding(self):
        cleaned = clean_data(self.sample_data)
        self.assertEqual(cleaned['gender'].iloc[0], 1)
        self.assertEqual(cleaned['gender'].iloc[1], 0)
        self.assertEqual(cleaned['Partner'].iloc[0], 1)
        self.assertEqual(cleaned['Partner'].iloc[1], 0)

    def test_clean_data_total_charges_numeric(self):
        cleaned = clean_data(self.sample_data)
        self.assertEqual(cleaned['TotalCharges'].iloc[0], 100.5)
        self.assertEqual(cleaned['TotalCharges'].iloc[1], 0.0) # Empty string filled with 0.0

    def test_encode_categorical(self):
        cleaned = clean_data(self.sample_data)
        encoded = encode_categorical(cleaned)
        # MultipleLines has 'No' and 'Yes'
        self.assertIn('MultipleLines_No', encoded.columns)
        self.assertIn('MultipleLines_Yes', encoded.columns)

if __name__ == '__main__':
    unittest.main()
