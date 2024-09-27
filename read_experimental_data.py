#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 13:50:13 2024

@author: carles
"""

#Reading files
import numpy as np
import h5py
from IPython.display import clear_output
import time

import chaospy

# Read all data from the experiment
data = []
with h5py.File('SDI_QRNG.hdf5', 'r') as f:
    for run in range(11):
        data += [np.array(f[f'/events/set{run+1}'])]
        print('\r'+f'Run:{run}\r',end="")
        
        
""" Compute a single datapoint from all gathered statistics """
sigle_datapoint = rng_scaling(1e7,data)
""" -------------------------------------------------------"""

""" 
Compute the scalability of our method with various 
sub-sets of samples form the gathered data 
"""

H_vec_nrounds = [] 
Hmin_vec_nrounds = []
H_QAEP_nrounds = []
H_GEAT_nrounds = []

H_vec_nrounds_var = [] 
Hmin_vec_nrounds_var = []
H_QAEP_nrounds_var = []
H_GEAT_nrounds_var = []

N = 30
ns_vec = np.linspace(3.0,7.0,N)

for i in range(N):
     
    start = time.process_time()
    
    ns = 10.0**(ns_vec[i])#N_total_data #Fix total amount of datapoints (for fs corrections)
    out = rng_scaling(ns,data)
    
    # Values
    H_vec_nrounds += [out[3][0]]
    Hmin_vec_nrounds += [out[2][0]]
    H_QAEP_nrounds += [out[4][0]]
    H_GEAT_nrounds += [out[6][0]]
    
    # Variances
    H_vec_nrounds_var += [out[3][1]]
    Hmin_vec_nrounds_var += [out[2][1]]
    H_QAEP_nrounds_var += [out[4][1]]
    H_GEAT_nrounds_var += [out[6][1]]
    
    end = time.process_time()
    print(f'Completed for {ns} samples')
    print(f'Process ended in {end-start} seconds')

    