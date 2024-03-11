import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def moving_average(data, window_size):
    """Smooth the time series data with a moving average."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

def identify_peaks(data, min_amplitude, min_distance):
    """Identify peaks in the data based on amplitude and distance."""
    peaks, _ = find_peaks(data, height=min_amplitude, distance=min_distance)
    return peaks


def cluster_peaks(peaks, eps, min_samples):
    """
    Cluster peaks using DBSCAN.

    Parameters:
    - peaks (numpy.ndarray): Array of peak indices.
    - eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
    - numpy.ndarray: Cluster labels for each peak.
    """
    if len(peaks) == 0:
        return np.array([-1])  # No peaks, return -1 as a label

    peaks_feature_vectors = np.column_stack((peaks, np.zeros_like(peaks)))
    scaler = StandardScaler()
    peaks_feature_vectors_scaled = scaler.fit_transform(peaks_feature_vectors)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(peaks_feature_vectors_scaled)
    return dbscan.labels_

def process_strain_data(strain_array, window_size=20, min_amplitude=-9.2, eps=0.05):
    """
    Process strain data for each spatial location.

    Parameters:
    - strain_array (numpy.ndarray): Array of strain values of shape (number_of_samples, spatial_locations).
    - window_size (int): Window size for moving average smoothing.
    - min_amplitude (float): Minimum amplitude for peak detection.
    - min_distance (int): Minimum distance between peaks for peak detection.
    - eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
    - numpy.ndarray: Array of new times for each spatial location.
    """
    num_samples, num_locations = strain_array.shape
    new_times_array = np.zeros((num_samples, num_locations))

    for location in range(num_locations):
        #print(f"location: {location}")

        data = strain_array[:, location]
    
        # Smooth the data with moving average
        smoothed_data = moving_average(data, window_size)

        # Identify peaks
        peaks = identify_peaks(smoothed_data, min_amplitude, 1)
       
     
        # Cluster peaks using DBSCAN
        cluster_labels = cluster_peaks(peaks, eps, 1)

        # Calculate average times for each cluster
        unique_clusters = np.unique(cluster_labels)
        for cluster_label in unique_clusters:
            if cluster_label != -1:
                cluster_peaks_indices = peaks[cluster_labels == cluster_label]
                average_time = np.mean(cluster_peaks_indices)
                new_times_array[int(average_time), location] = smoothed_data[int(average_time)]

    return new_times_array

  





############################# 2D #############################

import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.signal import convolve2d





def identify_peaks_2d(arr, threshold):
    
    rows, cols = arr.shape
    peak_rows, peak_cols = [], []

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if (arr[i, j] >= arr[i - 1, j] and
                arr[i, j] >= arr[i + 1, j] and
                arr[i, j] >= arr[i, j - 1] and
                arr[i, j] >= arr[i, j + 1] and
                arr[i, j] >= threshold):
                peak_rows.append(i)
                peak_cols.append(j)

    return np.array(list(zip(peak_rows, peak_cols)))

def cluster_peaks_2d(peaks, eps, min_samples):
    """
    Cluster peaks in 2D using DBSCAN.

    Parameters:
    - peaks (numpy.ndarray): Array of peak indices.
    - eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
    - numpy.ndarray: Cluster labels for each peak.
    """
    if len(peaks) == 0:
        return np.array([-1])  # No peaks, return -1 as a label

    peaks_feature_vectors = np.column_stack((peaks, np.zeros_like(peaks)))
    scaler = StandardScaler()
    peaks_feature_vectors_scaled = scaler.fit_transform(peaks_feature_vectors)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(peaks_feature_vectors_scaled)
    return dbscan.labels_


    

def process_strain_data_2d(strain_array, min_amplitude=-9.2, eps=0.05, window_size=10):
  

    num_samples, num_locations = strain_array.shape
    smoothed_data = np.empty_like(strain_array)
    for location in range(num_locations):
    
        # Smooth the data with moving average
        smoothed_data[:,location] = moving_average(strain_array[:,location], window_size)

    # Identify peaks in the smoothed data
    peaks = identify_peaks_2d(smoothed_data, min_amplitude)
    
    # Cluster peaks using 2D DBSCAN
    cluster_labels = cluster_peaks_2d(peaks, eps, 1)
    
    # Calculate average times for each cluster
    unique_clusters = np.unique(cluster_labels)
 

    new_times_array = np.zeros((num_samples, num_locations))

    for cluster_label in unique_clusters:
        if cluster_label != -1:
            
            cluster_peaks_indices = peaks[cluster_labels == cluster_label]
            
            average_time = np.mean(cluster_peaks_indices[:,0])
            average_channel = np.mean(cluster_peaks_indices[:,1])
            #for location in range(num_locations):
            new_times_array[int(average_time), int(average_channel)] = smoothed_data[int(average_time), int(average_channel)]

    return new_times_array



