import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

class AnomalyDetector:
    def __init__(self, data):
        """
        Initialize the AnomalyDetector object.

        Parameters:
        data (pandas Series): The input data for anomaly detection.
        """
        self.data = self.get_data(data)
        self.model = None


    def get_data(self, data):
        if isinstance(data, list):
            data = pd.Series(data)

            # Reshape the data if it's a 1D Series
            if data.ndim == 1:
                data = data.values.reshape(-1, 1)

        elif not isinstance(data, pd.Series):
            raise ValueError("Input data must be a pandas Series")

        return data


    def LOF_detector(self, threshold=0.005, n_neighbors=3):
        """
        Detect anomalies in the data using Local Outlier Factor.

        Parameters:
        threshold (float): The threshold for anomaly detection. Values below this threshold are considered anomalies.

        Returns:
        pandas.Series: A Series containing only the data points that are classified as anomalies.
        """

        # Initialize and fit the Local Outlier Factor model
        self.model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
        self.model.fit(self.data)

        # Predict anomaly scores (negative scores represent anomalies)
        anomaly_scores = -self.model.decision_function(self.data)
        
        # Create a Series indicating whether each data point is an anomaly or not
        anomalies = pd.Series(anomaly_scores > threshold, index=pd.RangeIndex(len(anomaly_scores)))

        # Filter the data to include only the anomalies
        anomalies_data = self.data[anomalies].flatten()

        return anomalies_data


    def IsolationForest_detector(self, threshold=0.005, n_estimators=100):
        """
        Detect anomalies in the data.

        Parameters:
        threshold (float): The threshold for anomaly detection. Values below this threshold are considered anomalies.

        Returns:
        pandas.Series: A Series containing only the data points that are classified as anomalies.
        """
        # Initialize and fit the Isolation Forest model
        self.model = IsolationForest(n_estimators=n_estimators,random_state=42)
        self.model.fit(self.data)

        # Predict anomaly scores
        anomaly_scores = self.model.decision_function(self.data)
        
        # Create a Series indicating whether each data point is an anomaly or not
        anomalies = pd.Series(anomaly_scores < threshold, index=pd.RangeIndex(len(anomaly_scores)))

        # Filter the data to include only the anomalies
        anomalies_data = self.data[anomalies].flatten()

        return anomalies_data
    

    def IQR_detector(self, threshold=1.5):
        """
        Detect anomalies in the data using Interquartile Range (IQR) method.

        Parameters:
        threshold (float): The threshold for anomaly detection. Values above or below this threshold are considered anomalies.

        Returns:
        pandas.Series: A Series containing only the data points that are classified as anomalies.
        """
        # Calculate quartiles
        Q1 = np.percentile(self.data, 25)
        Q3 = np.percentile(self.data, 75)
        
        # Calculate IQR
        IQR = Q3 - Q1
        
        # Define the lower and upper bounds
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Find outliers
        outliers = (self.data < lower_bound) | (self.data > upper_bound)
        
        return self.data[outliers].flatten()


    def percentile_detector(self, lower_percentile=5, upper_percentile=95):
        """
        Detect anomalies in the data using percentile-based method.

        Parameters:
        lower_percentile (float): The lower percentile threshold for anomaly detection.
        upper_percentile (float): The upper percentile threshold for anomaly detection.

        Returns:
        pandas.Series: A Series containing only the data points that are classified as anomalies.
        """
        lower_bound = np.percentile(self.data, lower_percentile)
        upper_bound = np.percentile(self.data, upper_percentile)
        
        # Find outliers
        outliers = (self.data < lower_bound) | (self.data > upper_bound)
        
        return self.data[outliers].flatten()
    

    def robust_covariance_detector(self, contamination=0.1):
        """
        Detect anomalies in the data using Robust Covariance.

        Parameters:
        contamination (float): The proportion of outliers to be detected.

        Returns:
        pandas.Series: A Series containing only the data points that are classified as anomalies.
        """
        # Initialize and fit the Robust Covariance model
        self.model = EllipticEnvelope(contamination=contamination)
        self.model.fit(self.data)
        
        # Predict outlier labels
        outliers = self.model.predict(self.data)
        
        # Filter the data to include only the anomalies
        anomalies_data = self.data[outliers == -1].flatten()
        
        return anomalies_data
    

    def knn_detector(self, n_neighbors=5, contamination=0.1):
        """
        Detect anomalies in the data using k-Nearest Neighbors (kNN).

        Parameters:
        n_neighbors (int): Number of neighbors to use for kNN.
        contamination (float): The proportion of outliers to be detected.

        Returns:
        pandas.Series: A Series containing only the data points that are classified as anomalies.
        """
        # Initialize and fit the kNN model
        self.model = NearestNeighbors(n_neighbors=n_neighbors)
        self.model.fit(self.data)

        # Find distances to the k nearest neighbors
        distances, _ = self.model.kneighbors()

        # Calculate the anomaly score as the mean distance to the k nearest neighbors
        anomaly_scores = np.mean(distances, axis=1)

        # Determine the threshold based on the contamination rate
        threshold = np.percentile(anomaly_scores, 100 * (1 - contamination))

        # Find outliers based on the threshold
        outliers = anomaly_scores > threshold

        # Filter the data to include only the anomalies
        anomalies_data = self.data[outliers].flatten()

        return anomalies_data
    
    def build_autoencoder(self, input_dim):
        """
        Build the autoencoder model.

        Parameters:
        input_dim (int): Dimension of the input data.

        Returns:
        keras.Model: The autoencoder model.
        """
        # Define the encoder
        encoder = keras.Sequential([
            keras.layers.Dense(32, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dense(16, activation='relu')
        ])

        # Define the decoder
        decoder = keras.Sequential([
            keras.layers.Dense(32, activation='relu', input_shape=(16,)),
            keras.layers.Dense(input_dim, activation='sigmoid')
        ])

        # Combine encoder and decoder to form autoencoder
        autoencoder = keras.Sequential([encoder, decoder])

        return autoencoder

    def autoencoder_detector(self, epochs=50, batch_size=32, test_size=0.2, random_state=42):
        """
        Detect anomalies in the data using an autoencoder.

        Parameters:
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed for reproducibility.

        Returns:
        numpy.ndarray: An array containing the data points that are classified as anomalies.
        """
        # Split data into train and test sets
        X_train, X_test = train_test_split(self.data, test_size=test_size, random_state=random_state)

        # Normalize data
        scaler = keras.preprocessing.MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Build and compile the autoencoder model
        input_dim = X_train_scaled.shape[1]
        autoencoder = self.build_autoencoder(input_dim)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        # Train the autoencoder
        autoencoder.fit(X_train_scaled, X_train_scaled, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(X_test_scaled, X_test_scaled), verbose=0)

        # Reconstruct data using the trained autoencoder
        reconstructions = autoencoder.predict(X_test_scaled)

        # Calculate reconstruction errors
        reconstruction_errors = np.mean(np.square(X_test_scaled - reconstructions), axis=1)

        # Determine threshold for anomaly detection (e.g., 95th percentile)
        threshold = np.percentile(reconstruction_errors, 95)

        # Find outliers based on the threshold
        outliers = X_test[reconstruction_errors > threshold].flatten()

        return outliers
    

    def plot_data_with_outliers(self, outliers):
        """
        Plot the original data with outliers marked.

        Parameters:
        outliers (numpy.ndarray): An array containing the outliers to be marked in the plot.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.data, 'bo', label='Original Data')
        plt.plot(outliers, 'ro', label='Outliers')
        plt.title('Original Data with Outliers Marked')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.show()


# Example usage:
# Example data with outliers
data_with_outliers = pd.Series([1, 2, 3, 100, 4, 5, 6, 200])

# Create AnomalyDetector object
detector = AnomalyDetector(data_with_outliers)

# Detect anomalies using autoencoder
anomalies_data = detector.autoencoder_detector()

# Plot original data with outliers marked
detector.plot_data_with_outliers(anomalies_data)
