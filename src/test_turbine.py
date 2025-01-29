import unittest
import pandas as pd
from unittest.mock import patch
from turbine import load_data, clean_data, calculate_statistics, identify_anomalies

class TestTurbineFunctions(unittest.TestCase):

    def setUp(self):
        """Set up sample data for testing."""
        self.sample_data = pd.DataFrame({
            'timestamp': ['2023-01-01 00:00:00', '2023-01-01 01:00:00', '2023-01-01 02:00:00'],
            'turbine_id': [1, 1, 2],
            'power_output': [100, 200, 150]
        })
        self.sample_data['timestamp'] = pd.to_datetime(self.sample_data['timestamp'])

        self.sample_statistics = pd.DataFrame({
            'turbine_id': [1, 2],
            'min_output': [100, 150],
            'max_output': [200, 150],
            'avg_output': [150.0, 150.0],
            'std_output': [70.71067811865476, float('nan')]  # Corrected std values
        })

    @patch('turbine.pd.read_csv')
    def test_load_data(self, mock_read_csv):
        """Test loading CSV files."""
        mock_read_csv.side_effect = [self.sample_data, self.sample_data]
        result = load_data(['mock_file_1.csv', 'mock_file_2.csv'])
        expected = pd.concat([self.sample_data, self.sample_data], ignore_index=True)
        pd.testing.assert_frame_equal(result, expected)

    def test_clean_data(self):
        """Test data cleaning."""
        cleaned_data, missing_before, missing_after = clean_data(self.sample_data)
        self.assertEqual(cleaned_data.shape[0], 3)
        self.assertEqual(missing_before, 0)
        self.assertEqual(missing_after, 0)

    def test_calculate_statistics(self):
        """Test statistics calculation."""
        summary = calculate_statistics(self.sample_data)
        pd.testing.assert_frame_equal(
            summary,
            self.sample_statistics,
            check_dtype=False,
            check_exact=False
        )

    def test_identify_anomalies(self):
        """Test anomaly detection."""
        anomalies = identify_anomalies(self.sample_data, self.sample_statistics)
        self.assertEqual(anomalies.shape[0], 0)

if __name__ == '__main__':
    unittest.main()
