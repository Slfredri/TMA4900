
import scipy.signal
from utils import map_point_between_ranges
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

strain_1000 = rms_plot[:,int(map_point_between_ranges(point = 1000, reverse = True))]
df = pd.DataFrame({'Time': times, 'Data': strain_1000})
window_size = 20

# Compute the moving average using rolling mean
df['Moving_Avg'] = df['Data'].rolling(window=window_size).mean()



# Plot all the smoothed data
plt.figure(figsize=(8, 6))
plt.plot(df['Time'], df['Data'], label='Original Data', alpha=0.5)
plt.plot(df['Time'], df['Moving_Avg'], label=f'Moving Average (window={window_size})')

plt.xlabel('Time')
plt.ylabel('Data')

plt.legend()
plt.title('Smoothing Techniques Comparison')
plt.show()


amp_thres = -9.2
# Identify peaks in the smoothed data
peaks, _ = find_peaks(df['Moving_Avg'], distance=1, height = amp_thres)  # Adjust distance as needed



feature_vectors = np.array(peaks).reshape(-1, 1)

# Standardize the feature vectors (important for DBSCAN)
scaler = StandardScaler()
feature_vectors_scaled = scaler.fit_transform(feature_vectors)

# Apply DBSCAN for clustering
eps = 0.05  # Adjust the neighborhood distance
min_samples = 1  # Adjust the minimum number of samples in a cluster
dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(feature_vectors_scaled)

# Assign cluster labels back to time indices
cluster_labels = dbscan.labels_

plt.figure(figsize=(8, 6))
plt.plot(df['Time'], df['Data'], label='Original Data', alpha=0.5)
plt.plot(df['Time'], df['Moving_Avg'], label=f'Moving Average (window={window_size})')

# Highlight clustered peaks
for cluster_label in np.unique(cluster_labels):
    if cluster_label == -1:
        # -1 indicates noise (outliers)
        plt.scatter(peaks[cluster_labels == cluster_label], 
                    np.zeros_like(peaks[cluster_labels == cluster_label]), 
                    label=f'Noise (Outliers)', c='red', marker='x')
    else:
        cluster_peaks = df.iloc[peaks[cluster_labels == cluster_label]]
        plt.scatter(cluster_peaks['Time'], cluster_peaks['Moving_Avg'], 
                    label=f'Cluster {cluster_label}')

plt.axhline(y=amp_thres, color='r', linestyle='--', label='Amplitude threshold')
plt.xlabel('Time')
plt.ylabel('Data')
plt.legend(loc='best')
plt.title('Original Data, Moving Average, and Clustered Peaks using DBSCAN')
plt.show()