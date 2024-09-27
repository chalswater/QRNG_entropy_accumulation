#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 13:28:17 2024

@author: carles
"""

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
        

ns = 1e7
n_resamplings = 11
n_bins = int(1e7/ns)

nBB = 4 #True number of outcomes
nB = 3 #Number of outcomes (when re-labling outcome b=3)
nX = 3 #Number of states
nS = 2 #Number of scenarios
xstar = 2 #Which state we get randomness from
px = [0.25,0.25,0.5]

epsilon = 1e-8 #EAT epsilon
epsilon_fs = 1e-8 #Finite-size effects epsilon

xstar = 2

alpha_vec = []
beta_vec = []
dc_vec = []

Hmin_datapoint_vec = []
H_datapoint_vec = []
AEP_datapoint_vec = []
EAT_datapoint_vec = []

p_vec = np.zeros((nB,nX,n_resamplings))

remaining_time = 0.0

# Gauss-Radau quadrature
m_in = 4
m = int(m_in*2)
distribution = chaospy.Uniform(lower=1e-3, upper=1)
t, w = chaospy.quadrature.radau(m_in,distribution,1.0)
t = t[0]

for nn in range(n_resamplings):
    
    lista = chunkify(data[nn], n_bins)
    #lista = chunkify([data[jj] for jj in range(11)], n_bins)
    
    start = time.process_time() #Monitor time
    clear_output(wait=True)
    
    out_Hmin = None
    tries = 0
    while out_Hmin == None:
        tries += 1
        if tries >= 10:
            print('ERROR: Hmin is always NONE!')
            break
        
        new_dataset = []
        for i in range(n_bins):
            ind = np.random.randint(0,n_bins,dtype=int)
            new_dataset += [lista[ind]]
        new_dataset = new_dataset[0]
            
        #count the number of events in the new dataset
        print('\r' + f'{round(float(nn)/float(n_resamplings)*100.0,3)}%, counting the number of events...\r',end="")
        n_x = np.zeros(nX)
        n_b_x = np.zeros((nBB,nX))
        n_tot = 0.0
        
        for x in range(nX):
            for b in range(nBB):
                n_b_x[b][x] += np.count_nonzero(np.all(new_dataset == [x,b], axis=1))
                n_tot += n_b_x[b][x]
                n_x[x] +=  n_b_x[b][x]
    
        #calculate the frequencies
        print('\r' + f'{round(float(nn)/float(n_resamplings)*100.0,3)}%, calculating frequencies...\r',end="")
        pbx_data_pre = np.zeros((nBB,nX))
        for b in range(nBB):
            for x in range(nX):
                if n_x[x] > 0.0:
                    pbx_data_pre[b][x] = float(n_b_x[b][x])/float(n_x[x])
                else:
                    pbx_data_pre[b][x] = 0.0
                    
        #Get the statistics from experimental results
        print('\r' + f'{round(float(nn)/float(n_resamplings)*100.0,3)}%, calculating statistics...\r',end="")
        pbx_data = np.zeros((nB,nX)) #Observed statistics from experimental data: re-labling b=3
    
        gg = [0.5,0.5,0.0]
        for b in range(nB):
            for x in range(nX):
                pbx_data[b][x] = pbx_data_pre[b][x] + gg[b]*pbx_data_pre[3][x]
    
        #Dark counts
        dc_data_0 = pbx_data_pre[3][0] + pbx_data_pre[1][0]
        dc_data_1 = pbx_data_pre[3][1] + pbx_data_pre[0][1]
        dc_avg = (dc_data_0+dc_data_1)/2.0
        clear_output(wait=True)
        print('\r' + f'{round(float(nn)/float(n_resamplings)*100.0,3)}%, Time: {round(remaining_time,1)}s \r',end="")
    
        #Overlap
        olap_data_0 = pbx_data_pre[2][0]/(1.0-dc_data_0)**2.0
        olap_data_1 = pbx_data_pre[2][1]/(1.0-dc_data_1)**2.0
    
        #Alpha
        alpha_data_0 = np.sqrt(-np.log(olap_data_0))
        alpha_data_1 = np.sqrt(-np.log(olap_data_1))
        alpha_avg = (alpha_data_0+alpha_data_1)/2.0 #Averaged alpha form the statistics in the experiment
    
        #Beta
        exp_beta_01 = (pbx_data_pre[2][2]-dc_avg**2.0)/(1.0-dc_avg)**2.0
        exp_beta_0 = exp_beta_01 + (pbx_data_pre[1][2]-dc_avg*(1.0-dc_avg))/(1.0-dc_avg)**2.0
        exp_beta_1 = exp_beta_01 + (pbx_data_pre[0][2]-dc_avg*(1.0-dc_avg))/(1.0-dc_avg)**2.0
    
        beta0_data = np.sqrt(-np.log(exp_beta_0))
        beta1_data = np.sqrt(-np.log(exp_beta_1))
        beta_avg = (beta0_data+beta1_data)/2.0 #Averaged beta form the statistics in the experiment
    
        #Targetted alphas on the experiment, assuming noise eta=0.06
        eta = 0.06 # eta -> 1- efficiency (sorry for reversed notation)
        target_alpha = alpha_avg/np.sqrt(1.0-eta)
        target_beta = beta_avg/np.sqrt(1.0-eta)
        clear_output(wait=True)
    
        #add the amplitudes and dc prob to a list
        alpha_vec += [target_alpha]
        beta_vec += [target_beta]
        dc_vec += [dc_avg]
        
        #------------------------------------------------------------------------------------
        #Calculate the Entropy --------------------------------------------------------------
        #------------------------------------------------------------------------------------
        clear_output(wait=True)
        print('\r' + f'{round(float(nn)/float(n_resamplings)*100.0,3)}%, Time: {round(remaining_time,1)}s \r',end="")
        print('\r' + f'{round(float(nn)/float(n_resamplings)*100.0,3)}%, Calculating Entropy...\r',end="")
        alpha = target_alpha 
        beta = target_beta 
    
        beta0 = beta
        beta1 = beta
    
        olap = np.exp(-alpha**2.0)
        d = olap
        D02 = np.exp((-(alpha)**2.0)/2.0) * np.exp((-np.abs(beta0)**2.0)/2.0) * np.exp((-np.abs(beta1)**2.0)/2.0) * np.exp(alpha*beta0)
        D12 = np.exp((-(alpha)**2.0)/2.0) * np.exp((-np.abs(beta0)**2.0)/2.0) * np.exp((-np.abs(beta1)**2.0)/2.0) * np.exp(alpha*beta1)
    
        #States 
        a = np.abs((D02+D12)/np.sqrt(2.0*(1.0+d)))**2.0 + np.abs((D02-D12)/np.sqrt(2.0*(1.0-d)))**2.0
        if a >= 1.0:
            a = 1.0
        if a <= 0.0:
            a = 0.0
    
        v_list = []
        v_list += [np.array([[np.sqrt((1.0+d)/2.0), np.sqrt((1.0-d)/2.0),0.0]])]
        v_list += [np.array([[np.sqrt((1.0+d)/2.0),-np.sqrt((1.0-d)/2.0),0.0]])]
        v_list += [np.array([[(D02+D12)/np.sqrt(2.0*(1.0+d)),(D02-D12)/np.sqrt(2.0*(1.0-d)),np.sqrt(1.0-a)]])]
    
        rho = []
        for state in v_list:
            rho += [np.kron(np.conjugate(np.transpose(state)),state)]
    
        out_Hmin = Hmin(rho,pbx_data,nX,nB,3,xstar)
        out_H = H(m-1,w,t,rho,pbx_data,nX,nB,3,xstar)
    
    Hmin_datapoint_vec += [out_Hmin]
    H_datapoint_vec += [out_H]
    
    for b in range(nB):
        for x in range(nX):
            p_vec[b][x][nn] = pbx_data[b][x]
    
    # Assymptotic Equipartiotion Property (non iid)
    pb = [sum([px[x]*pbx_data[b][x] for x in range(nX)]) for b in range(nB)]
    Hmax = 2.0*np.log(sum([np.sqrt(pb[b]) for b in range(nB)]))
    etaAEP = np.sqrt(2.0**(-out_Hmin)) + np.sqrt(2.0**Hmax) + 1.0
    delta = 4.0*np.log(etaAEP)*np.sqrt(np.log(2.0/epsilon**2.0))
    
    AEP_datapoint_vec += [out_H - delta/np.sqrt(ns)]

    clear_output(wait=True)
    print('\r' + f'{round(float(nn)/float(n_resamplings)*100.0,3)}%, Time: {round(remaining_time,1)}s\r',end="")
    
    #Code execution time monitoring variable
    end = time.process_time()
    remaining_time = (end-start)*(n_resamplings-m)