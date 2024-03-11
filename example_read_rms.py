#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 17:31:15 2023
script to read pre-computed rms strain and corresponding meta data
@author: keving
"""

import sys,glob 
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt


## constants
TIME_OFFSET_DAS_MINS = 2
TIME_OFFSET_DAS_SECONDS = 41
TIME_OFFSET_DAS = TIME_OFFSET_DAS_MINS*60 + TIME_OFFSET_DAS_SECONDS # time offset between recordings and UTC time in seconds
DX_PHY = 1.02; DEC_CHS=4;  # physical spatial sampling and channel decimation during acqisition
LEN_REC= 10     # length of original recordings in seconds, ! not length of rms files




def read_h5(filepath, labels_group=None, labels_data=None, verbose=True): 
    """
    read data and meta data from hdf5 files

    Parameters
    ----------
    filepath : str
        full path to file to read.
    labels_group : list of strs, optional
        labels of the groups within the hdf5 file. The default is None.
    labels_data : list of strs, optional
        labels of the datasets within the hdf5 file. The default is None.

    Returns
    -------
    list_data : list of np.arrays
        list of data arrays.
    info_all : dict
        dict containing subdicts for each group/dataset and general attributes.

    """
    
    if labels_group is None: 
        labels_group = ("rms","gaps"); 
    if labels_data is None: 
        labels_data= ('rms',)
    info_all={}
    with h5py.File(filepath, 'r') as f:
        #print("\nGEN attrs:\n")
        keys = list(f.attrs.keys())
        vals = list(f.attrs.values())
        info={}; info["labels_group"]=labels_group; info["labels_data"]=labels_data
        for k,key in enumerate(keys):
            if verbose:     print("{} = {}".format(key,vals[k]))
            info[key] = vals[k]
        info_all["gen"] = info
        #print(f'{k} = {f.attrs[k]}')
        
        list_data=[];
        for l,glabel in enumerate(labels_group): 
            #print("DEBUG: \n group: {}".format(glabel))
            #print(list(f[glabel].keys()))
            if len(list(f[glabel].keys()))>0:
                dlabel = list(f[glabel].keys())[0]; #print(f"data: {dlabel}")
                if dlabel is not None:
                    list_data.append(np.array(f[glabel].get(dlabel)))
            #if verbose:     print("DEBUG:", list(f[glabel].attrs.keys()))
            keys = list(f[glabel].attrs.keys())
            vals = list(f[glabel].attrs.values())
            dict_tmp={}
            for k,key in enumerate(keys):
                if verbose:     print("{} = {}".format(key,vals[k]))
                dict_tmp[key] = vals[k]
            info_all[glabel] = dict_tmp
        
    return list_data, info_all 

if __name__ == '__main__':

   
    # slfredri code
    path = r'\\markov.math.ntnu.no\work\slfredri\20210831\Chs03400_04600'
    filepaths = os.listdir(r'\\markov.math.ntnu.no\work\slfredri\20210831\Chs03400_04600')
    filepaths = [path+'/'+p for p in filepaths]

    #settings
    idx_file_load=100 # index file to load from file list
    
    #plot settings
    vmin_rms, vmax_rms = -10.5, None
    plot_log=1; log_thres=1e-12


     # load data and info from file
    list_data, info = read_h5(filepaths[idx_file_load], labels_group=("rms","gaps"), labels_data=("rms",), verbose=False)
    rms = list_data[0]
    print(f"list_data: {list_data[0].shape}")



    # extract information 
    dt_rms = info["rms"]["dt_rms"]  # temporal sampling interval of rms data
    time_utc = info["gen"]["time_utc"]  # time of first sample 
    timestamp_rec = info["gen"]["timestamp_rec"] # unix timestamp of first sample, ! recording time has offset to UTC time 
    dx = info["gen"]["dx"]                      # actual spatial sampling
    dec_chs_full = info["gen"]["dec_chs_full"]  # full channel decimation factor: dec_chs*ch_stack*dec_chs_extr
    chmin_abs, chmax_abs = [info["gen"][label] for label in ["chmin_abs","chmax_abs"]] # absolut channel limits 
    chmin_rel, chmax_rel = [info["gen"][label] for label in ["chmin_rel","chmax_rel"]] # relative channel limits 
    
    #create channel and time arrays
    channels_abs = np.arange(chmin_abs,chmax_abs+dec_chs_full,dec_chs_full)
    channels_rel = np.arange(chmin_rel,chmax_rel+1)
    times = np.arange(0,rms.shape[0]*dt_rms,dt_rms)
    
    if plot_log:
        rms_plot = np.log10(rms+log_thres); label_cbar='log(RMS Strain)'
    else: 
        rms_plot = rms.copy(); label_cbar='RMS Strain'
    
    #plotting
    figure=plt.figure(figsize=(10,8), dpi=150)
    mesh=plt.pcolormesh(channels_rel, times, rms_plot, cmap='viridis',vmin=vmin_rms, vmax=vmax_rms)
    cbar=plt.colorbar(mappable=mesh, label=label_cbar)
    plt.gca().invert_yaxis()
    plt.xlabel("channel_rel"); plt.ylabel("Time [s]")
    plt.title(f"{time_utc} UTC")
    plt.show()
    


























    """
    #input
    date=20210831 # acquisition date in YYYYMMDD format
    path_data  = f'../../BaneNOR/Data_pro/Sections/RMS/{date}/'
    chlabel = 'chs03400_04600' #'chs14600_15800' # chs03400_04600 # channels section to load
    
    #settings
    idx_file_load=100 # index file to load from file list
    
    #plot settings
    vmin_rms, vmax_rms = -10.5, None
    plot_log=1; log_thres=1e-12
    
    
    # create file list
    filepaths = sorted(glob.glob(path_data + '/{}/rms_*_{}_*.hdf5'.format(chlabel.capitalize(),chlabel)))


    # load data and info from file
    list_data, info = read_h5(filepaths[idx_file_load], labels_group=("rms","gaps"), labels_data=("rms",))
    rms = list_data[0]
    
    # extract information 
    dt_rms = info["rms"]["dt_rms"]  # temporal sampling interval of rms data
    time_utc = info["gen"]["time_utc"]  # time of first sample 
    timestamp_rec = info["gen"]["timestamp_rec"] # unix timestamp of first sample, ! recording time has offset to UTC time 
    dx = info["gen"]["dx"]                      # actual spatial sampling
    dec_chs_full = info["gen"]["dec_chs_full"]  # full channel decimation factor: dec_chs*ch_stack*dec_chs_extr
    chmin_abs, chmax_abs = [info["gen"][label] for label in ["chmin_abs","chmax_abs"]] # absolut channel limits 
    chmin_rel, chmax_rel = [info["gen"][label] for label in ["chmin_rel","chmax_rel"]] # relative channel limits 
    
    #create channel and time arrays
    channels_abs = np.arange(chmin_abs,chmax_abs+dec_chs_full,dec_chs_full)
    channels_rel = np.arange(chmin_rel,chmax_rel+1)
    times = np.arange(0,rms.shape[0]*dt_rms,dt_rms)
    
    if plot_log:
        rms_plot = np.log10(rms+log_thres); label_cbar='log(RMS Strain)'
    else: 
        rms_plot = rms.copy(); label_cbar='RMS Strain'
    
    #plotting
    figure=plt.figure(figsize=(10,8), dpi=150)
    mesh=plt.pcolormesh(channels_rel, times, rms_plot, cmap='viridis',vmin=vmin_rms, vmax=vmax_rms)
    cbar=plt.colorbar(mappable=mesh, label=label_cbar)
    plt.gca().invert_yaxis()
    plt.xlabel("channel_rel"); plt.ylabel("Time [s]")
    plt.title(f"{time_utc} UTC")
    plt.show()
    
    print("done")
    """