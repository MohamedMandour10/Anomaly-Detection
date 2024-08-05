<div align="center">
    <img width="745" alt="DICOM Viewer" src="https://github.com/user-attachments/assets/f314b3d3-7f6b-40dd-a4c4-f68104cad8e8">
</div>

# AnomalyDetector Class

The `AnomalyDetector` class is a Python class designed to detect anomalies in a given dataset using various anomaly detection algorithms. It provides a flexible and easy-to-use interface for detecting outliers in one-dimensional data.

## Understanding Outliers
Outliers are data points that deviate significantly from the rest of the data. They can arise due to various reasons such as measurement errors, data corruption, or genuine anomalies in the underlying process being observed. Outliers can distort statistical analyses and machine learning models, leading to inaccurate results and predictions if not properly handled.


## Class Methods

### 1. Initialization
- **Method Name:** `__init__(self, data)`
- **Description:** Initializes the `AnomalyDetector` object with the input data for anomaly detection.

### 2. LOF Detector
- **Method Name:** `LOF_detector(self, threshold=0.005, n_neighbors=3)`
- LOF calculates the local density deviation of a data point with respect to its neighbors. 
- Points with significantly lower density than their neighbors are considered anomalies.

### 3. Isolation Forest Detector
- **Method Name:** `IsolationForest_detector(self, threshold=0.005, n_estimators=100)`
- **Description:** Detects anomalies in the data using the Isolation Forest algorithm.
- Isolation Forest isolates anomalies by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of that feature.
- Anomalies are typically isolated in fewer splits, leading to shorter path lengths in the tree structure.
  <img src="https://media.licdn.com/dms/image/D4D08AQEiI2518F5qhg/croft-frontend-shrinkToFit1024/0/1700684114746?e=2147483647&v=beta&t=Ywr27fx5TbLmyCUXU_I01XpjVjw5uRlMODz7DuPEDfE" width="500" height="300" alt="Isolation forest">


### 4. Autoencoder Detector
- **Method Name:** `autoencoder_detector(self, epochs=50, batch_size=32, test_size=0.2, random_state=42)`
- An autoencoder is trained to reconstruct the input data.
- Anomalies are detected by comparing the reconstruction error (difference between input and output) for each data point to a threshold. Points with high reconstruction error are considered anomalies.
 <img src="https://miro.medium.com/v2/resize:fit:1400/1*DgkrcXtA0S87VL-aRIJ2Tg.png" width="900" height="400" alt="Autoencoder">
  
### 5. IQR Detector
- **Method Name:** `IQR_detector(self, threshold=1.5)`
- IQR calculates the range between the first quartile (Q1) and third quartile (Q3) of the data.
- Data points outside the range `[Q1 - k * IQR, Q3 + k * IQR]` are considered anomalies, where `k` is a user-defined threshold.
  
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*0MPDTLn8KoLApoFvI0P2vQ.png" width="500" alt="IQR">


### 6. Percentile Detector
- **Method Name:** `percentile_detector(self, lower_percentile=5, upper_percentile=95)`
- This method uses percentile-based thresholds to detect anomalies.
- Data points outside the specified lower and upper percentiles are classified as anomalies.


### 7. Robust Covariance Detector
- **Method Name:** `robust_covariance_detector(self, contamination=0.1)`
- Robust Covariance estimates the covariance matrix of the data, which is less sensitive to outliers compared to the standard covariance matrix.
- Anomalies are identified as points with low probability density under the estimated covariance model.


### 8. kNN Detector
- **Method Name:** `knn_detector(self, n_neighbors=5, contamination=0.1)`
- kNN computes the distance between each data point and its k nearest neighbors.
- Anomalies are identified as points with relatively large average distances to their k nearest neighbors.

  <img src="https://www.researchgate.net/publication/349412387/figure/fig12/AS:992643886637056@1613676153257/Anomaly-detection-using-a-KNN.png" width="500" alt="KNN">
  

### 9. Plot Data with Outliers
- **Method Name:** `plot_data_with_outliers(self, outliers)`
- **Description:** Plots the original data with outliers marked for visualization.

## Usage
**Installation**: To run the application, you need to have the required Python packages installed. You can create a virtual environment and install the necessary dependencies listed in the `requirements.txt` file.

   ```bash
   pip install -r requirements.txt
   ```

To use the `AnomalyDetector` class, instantiate an object with the input data, and then call the desired anomaly detection method.

**Example**:
```python
# Instantiate AnomalyDetector object
detector = AnomalyDetector(data)

# Detect anomalies using LOF algorithm
lof_anomalies = detector.LOF_detector()

# Plot data with outliers
detector.plot_data_with_outliers(lof_anomalies)
```


