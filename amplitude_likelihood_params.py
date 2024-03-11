import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from example_read_rms import read_h5
from utils import generate_timestamps
import numpy as np
import os
from utils import map_point_between_ranges
from scipy.stats import norm

data = """14:22:45 1
14:23:10 1
14:23:27 1
14:24:03 1
14:24:17 1
14:24:44 1
14:25:07 1
14:25:58 1
14:26:07 1
14:27:43 1
14:27:58 1
14:28:06 1
14:29:00 1
14:29:43 1
14:30:35 1
14:30:48 1
14:31:12 2
14:31:35 1
14:31:45 1
14:31:54 1
14:32:34 1
14:33:06 1
14:33:21 1
14:33:27 1
14:33:46 1
14:34:10 2
14:34:51 1
14:35:06 1
14:35:16 2

"""

# Create a dataframe
logged_df = pd.read_csv(StringIO(data), sep='\s+', header=None, names=['Time', 'Type'])
logged_df['Time'] = logged_df['Time']+'.000'
# Convert 'Time' column to datetime format
logged_df['Timestamp'] = pd.to_datetime('21-08-31 ' + logged_df['Time'], format='%y-%m-%d %H:%M:%S.%f', errors='coerce')

logged_df=logged_df.drop(columns=['Time'])
logged_df = logged_df[['Timestamp', 'Type']]


 # slfredri code
path = r'\\markov.math.ntnu.no\work\slfredri\20210831\Chs03400_04600'
filepaths = os.listdir(r'\\markov.math.ntnu.no\work\slfredri\20210831\Chs03400_04600')
filepaths = [path+'/'+p for p in filepaths]

#plot settings
vmin_rms, vmax_rms = -10.5, None
plot_log=1; log_thres=1e-12

idx0 = 88
idx1 = 89
idx2 = 90
idx3 = 91

list_data0, info0 = read_h5(filepaths[idx0], labels_group=("rms","gaps"), labels_data=("rms",), verbose=False)
rms0 = list_data0[0]
rms_plot0 = np.log10(rms0+log_thres); label_cbar='log(RMS Strain)'
dt_rms0 = info0["rms"]["dt_rms"]  # temporal sampling interval of rms data
time_utc0 = info0["gen"]["time_utc"]  # time of first sample 
times0 = generate_timestamps(time_utc0, dt_rms0, rms0.shape[0])

list_data1, info1 = read_h5(filepaths[idx1], labels_group=("rms","gaps"), labels_data=("rms",), verbose=False)
rms1 = list_data1[0]
rms_plot1 = np.log10(rms1+log_thres); label_cbar='log(RMS Strain)'
dt_rms1 = info1["rms"]["dt_rms"]  # temporal sampling interval of rms data
time_utc1 = info1["gen"]["time_utc"]  # time of first sample 
times1 = generate_timestamps(time_utc1, dt_rms1, rms1.shape[0])

list_data2, info2 = read_h5(filepaths[idx2], labels_group=("rms","gaps"), labels_data=("rms",), verbose=False)
rms2 = list_data2[0]
rms_plot2 = np.log10(rms2+log_thres); label_cbar='log(RMS Strain)'
dt_rms2 = info2["rms"]["dt_rms"]  # temporal sampling interval of rms data
time_utc2 = info2["gen"]["time_utc"]  # time of first sample 
times2 = generate_timestamps(time_utc2, dt_rms2, rms2.shape[0])

list_data3, info3 = read_h5(filepaths[idx3], labels_group=("rms","gaps"), labels_data=("rms",), verbose=False)
rms3 = list_data3[0]
rms_plot3 = np.log10(rms3+log_thres); label_cbar='log(RMS Strain)'
dt_rms3 = info3["rms"]["dt_rms"]  # temporal sampling interval of rms data
time_utc3 = info3["gen"]["time_utc"]  # time of first sample 
times3 = generate_timestamps(time_utc3, dt_rms3, rms3.shape[0])


# Indices
indices = [0,1,2,3]

# Lists to store data
list_datas = []
infos = []
rms_list = []
rms_plot_list = []
dt_rms_list = []
time_utc_list = []
times_list = []

# Process each index
for idx in indices:
    # Append data to respective lists
    list_datas.append(locals()[f'list_data{idx}'])
    infos.append(locals()[f'info{idx}'])
    rms_list.append(locals()[f'rms{idx}'])
    rms_plot_list.append(locals()[f'rms_plot{idx}'])
    dt_rms_list.append(locals()[f'dt_rms{idx}'])
    time_utc_list.append(locals()[f'time_utc{idx}'])
    times_list.append(locals()[f'times{idx}'])

    # Set label_cbar for each configuration
    label_cbar = 'log(RMS Strain)'




amplitude_df = pd.DataFrame({'Timestamp': [timestamp for sublist in times_list for timestamp in sublist],
                            'Amplitude': [amplitude for sublist in rms_plot_list for amplitude in sublist[:,int(map_point_between_ranges(1000, reverse=True))]]})

amplitude_df['Timestamp'] = amplitude_df['Timestamp'].dt.strftime('%y-%m-%d %H:%M:%S')
amplitude_df['Timestamp'] = pd.to_datetime(amplitude_df['Timestamp'], format='%y-%m-%d %H:%M:%S')
amplitude_df = amplitude_df[~amplitude_df.duplicated(subset='Timestamp',keep='first')]


DF = pd.merge(logged_df, amplitude_df, on='Timestamp', how='inner')




DF1 = DF[DF['Type']==1]
DF2 = DF[DF['Type']==2]

pi_0_1 = len(DF1)/len(DF) # Prior probability of being a car
pi_0_2 = len(DF2)/len(DF) # Prior probability of being a train

#pi_0_1 = 0.5
#pi_0_2 = 0.5

alpha_1 = np.mean(DF1['Amplitude'])
r_1 = np.std(DF1['Amplitude'])

alpha_2 = np.mean(DF2['Amplitude'])
r_2 = np.std(DF2['Amplitude'])

phi_1 = norm(loc=alpha_1, scale= r_1)
phi_2 = norm(loc=alpha_2, scale= r_2)

print(pi_0_1)
print(pi_0_2)
print(alpha_1)
print(alpha_2)
print(r_1)
print(r_2)