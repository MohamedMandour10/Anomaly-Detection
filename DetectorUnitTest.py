import unittest
import numpy as np
import pandas as pd
from AnomalyDetector import AnomalyDetector

class TestAnomalyDetector(unittest.TestCase):
    def setUp(self):
        # Generate larger and more complex synthetic data
        np.random.seed(42)
        num_samples = 1000
        normal_data = np.random.normal(loc=0, scale=1, size=num_samples)
        outliers = np.random.normal(loc=10, scale=5, size=50)  # Introduce outliers
        self.data_with_outliers = pd.Series(np.concatenate([normal_data, outliers]))
        self.detector = AnomalyDetector(self.data_with_outliers)

    def test_knn_detector_with_outliers(self):
        # Test kNN detector method with data containing outliers
        anomalies_data = self.detector.knn_detector()
        self.assertIsInstance(anomalies_data, np.ndarray)

    def test_autoencoder_detector_with_outliers(self):
        # Test autoencoder detector method with data containing outliers
        anomalies_data = self.detector.autoencoder_detector()
        self.assertIsInstance(anomalies_data, np.ndarray)

    def test_plot_data_with_outliers(self):
        # Test plot_data_with_outliers method
        self.detector.plot_data_with_outliers(self.data_with_outliers)
        # No assert statement since we're testing for any errors in the plotting function

    def test_knn_detector_invalid_parameters(self):
        # Test kNN detector method with invalid parameters
        with self.assertRaises(ValueError):
            self.detector.knn_detector(n_neighbors=-1)
        with self.assertRaises(ValueError):
            self.detector.knn_detector(contamination=-0.5)

    def test_autoencoder_detector_invalid_parameters(self):
        # Test autoencoder detector method with invalid parameters
        with self.assertRaises(ValueError):
            self.detector.autoencoder_detector(epochs=0)
        with self.assertRaises(ValueError):
            self.detector.autoencoder_detector(batch_size=0)
        with self.assertRaises(ValueError):
            self.detector.autoencoder_detector(test_size=-0.5)

    def test_plot_data_with_outliers_invalid_data(self):
        # Test plot_data_with_outliers method with invalid data
        with self.assertRaises(TypeError):
            self.detector.plot_data_with_outliers(None)

if __name__ == '__main__':
    unittest.main()
