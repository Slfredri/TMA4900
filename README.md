# TMA4900
Code for car and train tracking/classification at Selsbakk.

Most important files: 

data_processing.py - contains the algorithm that produces signal picks, using 2D DBSCAN clustering + moving average smoothing. 
tracker.py - contains a modified version of the stone soup JPDA tracker, to accommodate a restricted field of view, and classification of vehicles.
utils.py - contains general helper functions.

Some of the code is no longer in use, and in the notebooks one can explore how the functions work together, and they also contain code for producing figures.
