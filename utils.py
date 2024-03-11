import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from datetime import datetime, timedelta
import plotly.graph_objects as go
import numpy as np
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, truths=None, all_measurements=None, all_tracks=None, FOV=None):
        self.truth_pos_vectors = [[point.state_vector[0] for point in truth] for truth in truths] if truths is not None else None
        self.truth_vel_vectors = [[point.state_vector[1] for point in truth] for truth in truths] if truths is not None else None
        self.truth_timestamps = [[point.timestamp for point in truth] for truth in truths] if truths is not None else None

        self.measurement_state_vectors = [measurement.state_vector[0] for measurement in [measurement for measurement_set in all_measurements for measurement in measurement_set]] if all_measurements is not None else None
        self.measurement_timestamps = [measurement.timestamp for measurement in [measurement for measurement_set in all_measurements for measurement in measurement_set]] if all_measurements is not None else None

        self.track_pos_vectors = [[point.state_vector[0] for point in track] for track in all_tracks] if all_tracks is not None else None
        self.track_vel_vectors = [[point.state_vector[1] for point in track] for track in all_tracks] if all_tracks is not None else None
        self.track_timestamps = [[point.timestamp for point in track] for track in all_tracks] if all_tracks is not None else None
        self.track_pos_covar = [[point.covar[0, 0] for point in track.states] for track in all_tracks] if all_tracks is not None else None
        self.track_vel_covar = [[point.covar[1, 1] for point in track.states] for track in all_tracks] if all_tracks is not None else None

        self.FOV = FOV
        #self.dx = 4.085200763098726
        self.dx = 1

    def plot(self, plot_truths=False, plot_measurements=False, plot_tracks=False, plot_pos=True, plotly=False):
        if plotly:
            return self.plot_plotly(plot_truths, plot_measurements, plot_tracks, plot_pos)
        

    def plot_plotly(self, plot_truths=False, plot_measurements=False, plot_tracks=False, plot_pos=True):
        

        fig = go.Figure(layout=dict(width=800, height=600))
        
        #######################################################################################################
        if plot_truths:
            if plot_pos:
                
                for i, (data1, data2) in enumerate(zip(self.truth_pos_vectors, self.truth_timestamps)):
                    show_legend = i == 0
                    fig.add_trace(go.Scatter(x=[self.dx*el for el in data1], y=data2, mode='lines',marker=dict(color='blue'), name='Ground truth' if i == 0 else '', showlegend=show_legend))
                

                fig.add_shape(
                    type="line",
                    x0=self.dx*self.FOV[0],
                    y0=min(self.measurement_timestamps),
                    x1=self.dx*self.FOV[0],
                    y1=max(self.measurement_timestamps),
                    line=dict(color="red", width=2)
                )

                fig.add_shape(
                    type="line",
                    x0=self.dx*self.FOV[1],
                    y0=min(self.measurement_timestamps),
                    x1=self.dx*self.FOV[1],
                    y1=max(self.measurement_timestamps),
                    line=dict(color="red", width=2)
                )
                
                fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', marker=dict(color='red'), name='Field of View'))
                fig.update_layout(title='Simulated ground truth positions',
                                  xaxis_title='Position',
                                  yaxis_title='Time',
                                  showlegend=True)
            else:
                for i, (data1, data2) in enumerate(zip(self.truth_vel_vectors, self.truth_timestamps)):
                    show_legend = i == 0
                    
                    fig.add_trace(go.Scatter(x=[self.dx*el for el in data1], y=data2, mode='lines', marker=dict(color='blue'), name='Ground truth' if i == 0 else '', showlegend=show_legend))
                fig.update_layout(title='Simulated ground truth velocities',
                                  xaxis_title='Velocity',
                                  yaxis_title='Time',
                                  showlegend=True)
        #####################################################################################################
        if plot_measurements:
            try:
                for i, (data1, data2) in enumerate(zip(self.truth_pos_vectors, self.truth_timestamps)):
                    show_legend = i == 0
                    fig.add_trace(go.Scatter(x=[self.dx*el for el in data1], y=data2, mode='lines',marker=dict(color='blue'), name='Ground truth' if i == 0 else '', showlegend=show_legend))
                
                fig.update_layout(title='Simulated ground truth positions',
                                  xaxis_title='Position',
                                  yaxis_title='Time',
                                  showlegend=True)
            except:
                pass
            
            
            fig.add_shape(
            type="line",
            x0=self.dx*self.FOV[0],
            y0=min(self.measurement_timestamps),
            x1=self.dx*self.FOV[0],
            y1=max(self.measurement_timestamps),
            line=dict(color="red", width=2)
        )

            fig.add_shape(
                type="line",
                x0=self.dx*self.FOV[1],
                y0=min(self.measurement_timestamps),
                x1=self.dx*self.FOV[1],
                y1=max(self.measurement_timestamps),
                line=dict(color="red", width=2),
                showlegend = False
            )
            
            fig.add_trace(go.Scatter(x=[self.dx*el for el in self.measurement_state_vectors], y=self.measurement_timestamps, mode='markers', marker=dict(symbol='x', color = 'green'), name='Measurements'))
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', marker=dict(color='red'), name='Field of View'))
            fig.update_layout(title='Measurements',
                                  xaxis_title='Position',
                                  yaxis_title='Time',
                                  showlegend=True)

        ################################################################################################################
        if plot_tracks:
            if plot_pos:
                try:
                    for i, (data1, data2) in enumerate(zip(self.truth_pos_vectors, self.truth_timestamps)):
                        show_legend = i == 0
                        fig.add_trace(go.Scatter(x=[self.dx*el for el in data1], y=data2,mode='lines', marker=dict(color='blue'), name='Ground truth' if i == 0 else '', showlegend=show_legend))
                    fig.update_layout(title='Simulated ground truth positions',
                                  xaxis_title='Position',
                                  yaxis_title='Time',
                                  showlegend=True)
                except:
                    pass
                fig.add_shape(
                    type="line",
                    x0=self.dx*self.FOV[0],
                    y0=min(self.measurement_timestamps),
                    x1=self.dx*self.FOV[0],
                    y1=max(self.measurement_timestamps),
                    line=dict(color="red", width=2)
                )

                fig.add_shape(
                    type="line",
                    x0=self.dx*self.FOV[1],
                    y0=min(self.measurement_timestamps),
                    x1=self.dx*self.FOV[1],
                    y1=max(self.measurement_timestamps),
                    line=dict(color="red", width=2)
                )
                
                fig.add_trace(go.Scatter(x=[self.dx*el for el in self.measurement_state_vectors], y=self.measurement_timestamps, mode='markers', marker=dict(symbol='x', color = 'green'), name='Measurements'))
                
                cmap = get_cmap('tab20')
                for i, (data1, data2) in enumerate(zip(self.track_pos_vectors, self.track_timestamps)):
                    color = cmap(i % 20)
                    color = f'rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},{color[3]})'
                    
                    # Conditionally set showlegend based on i
                    show_legend = True if i == 0 else False
                    
                    fig.add_trace(go.Scatter(x=[self.dx*a - 2 * np.sqrt(self.dx*b) for a, b in zip(data1, self.track_pos_covar[i])], y=data2, mode='lines', line=dict(color=color, dash='dash'), name='Uncertainty' if show_legend else '', showlegend=show_legend))
                    fig.add_trace(go.Scatter(x=[self.dx*el for el in data1], y=data2, mode='lines+markers', line=dict(color=color), name=f'Track {i+1}', showlegend=True))
                    
                    fig.add_trace(go.Scatter(x=[self.dx*a + 2 * np.sqrt(self.dx*b) for a, b in zip(data1, self.track_pos_covar[i])], y=data2, mode='lines', line=dict(color=color, dash='dash'), showlegend=False))
                
                # Add a dummy scatter trace with custom legend entry for Field of View
                fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', marker=dict(color='red'), name='Field of View'))
                
                fig.update_layout(title='Estimated tracks, measurements and simulated ground truth positions',
                                xaxis_title='Position',
                                yaxis_title='Time',
                                showlegend=True)

            else:
                try:
                    for i, (data1, data2) in enumerate(zip(self.truth_vel_vectors, self.truth_timestamps)):
                        
                        show_legend = i == 0
                        fig.add_trace(go.Scatter(x=[self.dx*el for el in data1], y=data2, mode='lines',marker=dict(color='blue'), name='Ground truth' if i == 0 else '', showlegend=show_legend))
                    fig.update_layout(title='Simulated ground truth velocities',
                                  xaxis_title='Position',
                                  yaxis_title='Time',
                                  showlegend=True)

                except:
                
                    pass
                cmap = get_cmap('tab20')
                for i, (data1, data2) in enumerate(zip(self.track_vel_vectors, self.track_timestamps)):
                    
                    show_legend = i == 0
                    color = cmap(i % 20)
                    color = f'rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},{color[3]})'

                    fig.add_trace(go.Scatter(x=[self.dx*el for el in data1], y=data2, mode='lines+markers', line=dict(color=color), name='Track' if i == 0 else '',showlegend = show_legend))
                    fig.add_trace(go.Scatter(x=[self.dx*a - 2 * np.sqrt(self.dx*b) for a, b in zip(data1, self.track_vel_covar[i])], y=data2, mode='lines', line=dict(color=color, dash='dash'), name='Uncertainty' if i == 0 else '',showlegend = show_legend))
                    fig.add_trace(go.Scatter(x=[self.dx*a + 2 * np.sqrt(self.dx*b) for a, b in zip(data1, self.track_vel_covar[i])], y=data2, mode='lines', line=dict(color=color, dash='dash'), showlegend=False))
                fig.update_layout(title='Estimated velocities and simulated ground truth velocities',
                                xaxis_title='Velocity',
                                yaxis_title='Time',
                                showlegend=True)
        fig.show()

    def plot_groundtruth_pos(self, plotly=False):
        self.plot(plot_truths=True, plotly=plotly)

    def plot_groundtruth_vel(self, plotly=False):
        self.plot(plot_truths=True, plot_pos=False, plotly=plotly)

    def plot_measurements(self, plotly=False):
        self.plot(plot_measurements=True, plotly=plotly)

    def plot_track_pos(self, plotly=False):
        self.plot(plot_tracks=True, plotly=plotly)

    def plot_track_vel(self, plotly=False):
        self.plot(plot_tracks=True, plot_pos=False, plotly=plotly)




########################################################################################################################


class SimPlotter:
    def __init__(self, truths=None, all_measurements=None, all_tracks=None, FOV=None):
        self.truth_pos_vectors = [[point.state_vector[0] for point in truth] for truth in truths] if truths is not None else None
        self.truth_vel_vectors = [[point.state_vector[1] for point in truth] for truth in truths] if truths is not None else None
        self.truth_timestamps = [[point.timestamp for point in truth] for truth in truths] if truths is not None else None

        self.measurement_state_vectors = [measurement.state_vector[0] for measurement in [measurement for measurement_set in all_measurements for measurement in measurement_set]] if all_measurements is not None else None
        self.measurement_timestamps = [measurement.timestamp for measurement in [measurement for measurement_set in all_measurements for measurement in measurement_set]] if all_measurements is not None else None

        self.track_pos_vectors = [[point.state_vector[0] for point in track] for track in all_tracks] if all_tracks is not None else None
        self.track_vel_vectors = [[point.state_vector[1] for point in track] for track in all_tracks] if all_tracks is not None else None
        self.track_timestamps = [[point.timestamp for point in track] for track in all_tracks] if all_tracks is not None else None
        self.track_pos_covar = [[point.covar[0, 0] for point in track.states] for track in all_tracks] if all_tracks is not None else None
        self.track_vel_covar = [[point.covar[1, 1] for point in track.states] for track in all_tracks] if all_tracks is not None else None

        self.FOV = FOV

    def plot(self, plot_truths=False, plot_measurements=False, plot_tracks=False, plot_pos=True, plotly=False):
        if plotly:
            return self.plot_plotly(plot_truths, plot_measurements, plot_tracks, plot_pos)
        

    def plot_plotly(self, plot_truths=False, plot_measurements=False, plot_tracks=False, plot_pos=True):
        

        fig = go.Figure(layout=dict(width=800, height=600))
        
        #######################################################################################################
        if plot_truths:
            if plot_pos:
                
                for i, (data1, data2) in enumerate(zip(self.truth_pos_vectors, self.truth_timestamps)):
                    show_legend = i == 0
                    fig.add_trace(go.Scatter(x=[el for el in data1], y=data2, mode='lines',marker=dict(color='blue'), name='Ground truth' if i == 0 else '', showlegend=show_legend))
                

                fig.add_shape(
                    type="line",
                    x0=self.FOV[0],
                    y0=min(self.measurement_timestamps),
                    x1=self.FOV[0],
                    y1=max(self.measurement_timestamps),
                    line=dict(color="red", width=2)
                )

                fig.add_shape(
                    type="line",
                    x0=self.FOV[1],
                    y0=min(self.measurement_timestamps),
                    x1=self.FOV[1],
                    y1=max(self.measurement_timestamps),
                    line=dict(color="red", width=2)
                )
                
                fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', marker=dict(color='red'), name='Field of View'))
                fig.update_layout(title='Simulated ground truth positions',
                                  xaxis_title='Position',
                                  yaxis_title='Time',
                                  showlegend=True)
            else:
                for i, (data1, data2) in enumerate(zip(self.truth_vel_vectors, self.truth_timestamps)):
                    show_legend = i == 0
                    
                    fig.add_trace(go.Scatter(x=[el for el in data1], y=data2, mode='lines', marker=dict(color='blue'), name='Ground truth' if i == 0 else '', showlegend=show_legend))
                fig.update_layout(title='Simulated ground truth velocities',
                                  xaxis_title='Velocity',
                                  yaxis_title='Time',
                                  showlegend=True)
        #####################################################################################################
        if plot_measurements:
            try:
                for i, (data1, data2) in enumerate(zip(self.truth_pos_vectors, self.truth_timestamps)):
                    show_legend = i == 0
                    fig.add_trace(go.Scatter(x=[el for el in data1], y=data2, mode='lines',marker=dict(color='blue'), name='Ground truth' if i == 0 else '', showlegend=show_legend))
                
                fig.update_layout(title='Simulated ground truth positions',
                                  xaxis_title='Position',
                                  yaxis_title='Time',
                                  showlegend=True)
            except:
                pass
            
            
            fig.add_shape(
            type="line",
            x0=self.FOV[0],
            y0=min(self.measurement_timestamps),
            x1=self.FOV[0],
            y1=max(self.measurement_timestamps),
            line=dict(color="red", width=2)
        )

            fig.add_shape(
                type="line",
                x0=self.FOV[1],
                y0=min(self.measurement_timestamps),
                x1=self.FOV[1],
                y1=max(self.measurement_timestamps),
                line=dict(color="red", width=2),
                showlegend = False
            )
            
            fig.add_trace(go.Scatter(x=[el for el in self.measurement_state_vectors], y=self.measurement_timestamps, mode='markers', marker=dict(symbol='x', color = 'green'), name='Measurements'))
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', marker=dict(color='red'), name='Field of View'))
            fig.update_layout(title='Measurements',
                                  xaxis_title='Position',
                                  yaxis_title='Time',
                                  showlegend=True)

        ################################################################################################################
        if plot_tracks:
            if plot_pos:
                try:
                    for i, (data1, data2) in enumerate(zip(self.truth_pos_vectors, self.truth_timestamps)):
                        show_legend = i == 0
                        fig.add_trace(go.Scatter(x=[el for el in data1], y=data2,mode='lines', marker=dict(color='blue'), name='Ground truth' if i == 0 else '', showlegend=show_legend))
                    fig.update_layout(title='Simulated ground truth positions',
                                  xaxis_title='Position',
                                  yaxis_title='Time',
                                  showlegend=True)
                except:
                    pass
                fig.add_shape(
                    type="line",
                    x0=self.FOV[0],
                    y0=min(self.measurement_timestamps),
                    x1=self.FOV[0],
                    y1=max(self.measurement_timestamps),
                    line=dict(color="red", width=2)
                )

                fig.add_shape(
                    type="line",
                    x0=self.FOV[1],
                    y0=min(self.measurement_timestamps),
                    x1=self.FOV[1],
                    y1=max(self.measurement_timestamps),
                    line=dict(color="red", width=2)
                )
                
                fig.add_trace(go.Scatter(x=[el for el in self.measurement_state_vectors], y=self.measurement_timestamps, mode='markers', marker=dict(symbol='x', color = 'green'), name='Measurements'))
                
                cmap = get_cmap('tab20')
                for i, (data1, data2) in enumerate(zip(self.track_pos_vectors, self.track_timestamps)):
                    color = cmap(i % 20)
                    color = f'rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},{color[3]})'
                    
                    # Conditionally set showlegend based on i
                    show_legend = True if i == 0 else False
                    
                    fig.add_trace(go.Scatter(x=[a - 2 * np.sqrt(b) for a, b in zip(data1, self.track_pos_covar[i])], y=data2, mode='lines', line=dict(color=color, dash='dash'), name='Uncertainty' if show_legend else '', showlegend=show_legend))
                    fig.add_trace(go.Scatter(x=[el for el in data1], y=data2, mode='lines+markers', line=dict(color=color), name=f'Track {i+1}', showlegend=True))
                    
                    fig.add_trace(go.Scatter(x=[a + 2 * np.sqrt(b) for a, b in zip(data1, self.track_pos_covar[i])], y=data2, mode='lines', line=dict(color=color, dash='dash'), showlegend=False))
                
                # Add a dummy scatter trace with custom legend entry for Field of View
                fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', marker=dict(color='red'), name='Field of View'))
                
                fig.update_layout(title='Estimated tracks, measurements and simulated ground truth positions',
                                xaxis_title='Position',
                                yaxis_title='Time',
                                showlegend=True)

            else:
                try:
                    for i, (data1, data2) in enumerate(zip(self.truth_vel_vectors, self.truth_timestamps)):
                        
                        show_legend = i == 0
                        fig.add_trace(go.Scatter(x=[el for el in data1], y=data2, mode='lines',marker=dict(color='blue'), name='Ground truth' if i == 0 else '', showlegend=show_legend))
                    fig.update_layout(title='Simulated ground truth velocities',
                                  xaxis_title='Position',
                                  yaxis_title='Time',
                                  showlegend=True)

                except:
                
                    pass
                cmap = get_cmap('tab20')
                for i, (data1, data2) in enumerate(zip(self.track_vel_vectors, self.track_timestamps)):
                    
                    show_legend = i == 0
                    color = cmap(i % 20)
                    color = f'rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},{color[3]})'

                    fig.add_trace(go.Scatter(x=[el for el in data1], y=data2, mode='lines+markers', line=dict(color=color), name='Track' if i == 0 else '',showlegend = show_legend))
                    fig.add_trace(go.Scatter(x=[a - 2 * np.sqrt(b) for a, b in zip(data1, self.track_vel_covar[i])], y=data2, mode='lines', line=dict(color=color, dash='dash'), name='Uncertainty' if i == 0 else '',showlegend = show_legend))
                    fig.add_trace(go.Scatter(x=[a + 2 * np.sqrt(b) for a, b in zip(data1, self.track_vel_covar[i])], y=data2, mode='lines', line=dict(color=color, dash='dash'), showlegend=False))
                fig.update_layout(title='Estimated velocities and simulated ground truth velocities',
                                xaxis_title='Velocity',
                                yaxis_title='Time',
                                showlegend=True)
        fig.show()

    def plot_groundtruth_pos(self, plotly=False):
        self.plot(plot_truths=True, plotly=plotly)

    def plot_groundtruth_vel(self, plotly=False):
        self.plot(plot_truths=True, plot_pos=False, plotly=plotly)

    def plot_measurements(self, plotly=False):
        self.plot(plot_measurements=True, plotly=plotly)

    def plot_track_pos(self, plotly=False):
        self.plot(plot_tracks=True, plotly=plotly)

    def plot_track_vel(self, plotly=False):
        self.plot(plot_tracks=True, plot_pos=False, plotly=plotly)


# Function to calculate inward velocity vector towards FOV center
def inward_velocity_vector(position, fov_center, vel_range=5):
    # Calculate the vector pointing from the position to the FOV center
    direction_vector = fov_center - position
    # Normalize the direction vector
    norm = np.linalg.norm(direction_vector)
    return direction_vector / norm * vel_range


# Function to generate random position at the edges of the FOV
def random_edge_position(FOV):
    edge = np.random.choice([0, 1])  # 0: left, 1: right
    if edge == 0:
        return FOV[0]
    else:
        return FOV[1]





def map_point_between_ranges(point, source_range=(0, 300), target_range=(850, 1150), reverse=False):
    """
    Maps a point between source and target ranges using linear interpolation.

    Args:
        point (float): The point to be mapped.
        source_range (tuple): The source range (min, max).
        target_range (tuple): The target range (min, max).
        reverse (bool): If True, maps the point from target range to source range.

    Returns:
        float: The mapped point in the target or source range based on the 'reverse' argument.
    """
    if not reverse:
        source_min, source_max = source_range
        target_min, target_max = target_range

        # Perform linear interpolation
        mapped_point = ((point - source_min) / (source_max - source_min)) * (target_max - target_min) + target_min
        return mapped_point
    else:
        source_min, source_max = source_range
        target_min, target_max = target_range

        # Perform linear interpolation in reverse
        mapped_point = ((point - target_min) / (target_max - target_min)) * (source_max - source_min) + source_min
        return mapped_point





def generate_timestamps(start_time_utc, step_size_seconds, num_timestamps):
    # Convert start time in UTC format to a datetime object
    start_datetime = datetime.strptime(start_time_utc, '%Y-%m-%dT%H:%M:%S')

    # Initialize a list to store the timestamps
    times = []

    # Generate timestamps with the specified time step size (in seconds)
    for i in range(int(num_timestamps)):
        # Add the current time step to the start time
        current_time = start_datetime + timedelta(seconds=i * step_size_seconds)
        # Append the timestamp to the list
        times.append(current_time)

    return times

def count_points_between(start_time, step_size, end_time):
    current_time = start_time
    count = 0

    while current_time <= end_time:
        count += 1
        current_time += step_size

    return count



import itertools

# Define a distance metric between two timestamps (e.g., absolute difference in seconds)
def timestamp_distance(timestamp1, timestamp2):
    return abs(timestamp1 - timestamp2)

# Calculate the Hausdorff distance between two lists of timestamps
def hausdorff_distance(timestamps1, timestamps2, alpha = 0.1):
    max_distance = 0
    penalty_term = alpha * abs((len(timestamps1) - len(timestamps2)))
    for t1 in timestamps1:
        min_distance = float('inf')
        for t2 in timestamps2:
            d = timestamp_distance(t1, t2)
            min_distance = min(min_distance, d)
        max_distance = max(max_distance, min_distance)

    for t2 in timestamps2:
        min_distance = float('inf')
        for t1 in timestamps1:
            d = timestamp_distance(t2, t1)
            min_distance = min(min_distance, d)
        max_distance = max(max_distance, min_distance)

    return max_distance + penalty_term





import pandas as pd
from stonesoup.types.update import GaussianStateUpdate

def create_track_dataframe(all_tracks,tracker, update_thres=10):
    filtered_tracks = [track for track in all_tracks if sum(isinstance(state, GaussianStateUpdate) for state in track) >= update_thres]
    #num_updates = [sum(isinstance(state, GaussianStateUpdate) for state in track) for track in all_tracks]
    track_pos_vectors = [[point.state_vector[0] for point in track] for track in filtered_tracks] 
    track_vel_vectors = [[3.6*point.state_vector[1] for point in track] for track in filtered_tracks] 
    track_timestamps = [[point.timestamp for point in track] for track in filtered_tracks] 
    track_pos_covar = [[np.sqrt(point.covar[0, 0]) for point in track] for track in filtered_tracks] 
    track_vel_covar = [[3.6*np.sqrt(point.covar[1, 1]) for point in track] for track in filtered_tracks] 
    track_pos_lower = [[(a - 2 * b) for a, b in zip(track_pos_vectors[i], track_pos_covar[i])] for i in range(len(track_pos_vectors))]
    track_pos_upper = [[(a + 2 * b) for a, b in zip(track_pos_vectors[i], track_pos_covar[i])] for i in range(len(track_pos_vectors))]
    track_vel_lower = [[(a - 2 * b) for a, b in zip(track_vel_vectors[i], track_vel_covar[i])] for i in range(len(track_vel_vectors))]
    track_vel_upper = [[(a + 2 * b) for a, b in zip(track_vel_vectors[i], track_vel_covar[i])] for i in range(len(track_vel_vectors))]
    track_ids = [track.id for track in filtered_tracks]
    data1 = []
    
    for idx, (track_id, (pos, vel, timestamps, pos_covar, vel_covar, pos_lower, pos_upper, vel_lower, vel_upper)) in enumerate(zip(track_ids, zip(track_pos_vectors, track_vel_vectors, track_timestamps, track_pos_covar, track_vel_covar, track_pos_lower, track_pos_upper, track_vel_lower, track_vel_upper))):
       
        for i in range(len(pos)):
            data1.append([timestamps[i], idx, track_id, pos[i], pos_covar[i], vel[i], vel_covar[i], pos_lower[i], pos_upper[i], vel_lower[i], vel_upper[i]])

    df1 = pd.DataFrame(data1, columns=['timestamp', 'vehicle index', 'track ID', 'position [m]', 'std position', 'velocity [km/h]', 'std velocity', 'pos lower', 'pos upper', 'vel lower', 'vel upper'])

    data2 = []
    for track_id in track_ids:
        car_prob,train_prob, timestamps = tracker.all_probabilities[track_id]['car_probabilities'], tracker.all_probabilities[track_id]['train_probabilities'], tracker.all_probabilities_timestamps[track_id]
        for i in range(len(car_prob)):
            data2.append([timestamps[i], track_id, car_prob[i], train_prob[i]])
    df2 = pd.DataFrame(data2, columns=['timestamp', 'track ID', 'car probability', 'train probability'])        

    df = pd.merge(df1, df2, on= ['timestamp','track ID'], how='outer')
    return df




