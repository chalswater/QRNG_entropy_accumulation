#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 16:47:16 2024

@author: carles
"""

import math
import numpy as np                             
import mosek     

from cvxpy import *
import cvxpy as cp

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

import time
    
from qutip import *

from matplotlib import gridspec
import matplotlib.patches as patches

import chaospy

N = 100

nBB = 4 #True number of outcomes
nB = 3 #Number of outcomes (when re-labling outcome b=3)
nX = 3 #Number of states
nS = 2 #Number of scenarios
xstar = 2 #Which state we get randomness from

epsilon = 1e-8 #EAT epsilon
epsilon_fs = 1e-8 #Finite-size effects epsilon

#------------------------------------------------------------------------------
#                           COLOR MAP 
#------------------------------------------------------------------------------

#Colormap of the theoretical predictions around targetted alpha and beta

H_arr = np.zeros((N,N))
Hmin_arr = np.zeros((N,N))
Hmin_sdi_arr = np.zeros((N,N))
a_arr = np.zeros((N,N))

a_vec = np.linspace(0.01,1.0,N) #Vector of alphas
b_vec = np.linspace(0.3,0.9,N) #Vector of betas

eta = 0.06
dc = 4.485615570543283e-06

# Gauss-Radau quadrature
m_in = 4
m = int(m_in*2)
distribution = chaospy.Uniform(lower=1e-3, upper=1)
t, w = chaospy.quadrature.radau(m_in,distribution,1.0)
t = t[0]

for i in range(N):
    
    for j in range(N):

        alpha = a_vec[i]
        beta0 = b_vec[j]
        beta1 = b_vec[j]

        # States: |x=0 > = |a 0 > ; |x=1 > = |0 a > ; |x=1 > = |b0 b1 > 

        # Declare the overlaps of the states
        olap = np.exp(-alpha**2.0)
        D02 = np.exp((-(alpha)**2.0)/2.0) * np.exp((-np.abs(beta0)**2.0)/2.0) * np.exp((-np.abs(beta1)**2.0)/2.0) * np.exp(alpha*beta0)
        D12 = np.exp((-(alpha)**2.0)/2.0) * np.exp((-np.abs(beta0)**2.0)/2.0) * np.exp((-np.abs(beta1)**2.0)/2.0) * np.exp(alpha*beta1) 
        d = olap
        
        # Apply noise to amplitudes
        alpha = np.sqrt(1.0-eta)*alpha
        beta0 = np.sqrt(1.0-eta)*beta0
        beta1 = np.sqrt(1.0-eta)*beta1

        # Write up all the observed probabilities pbx[b][x]
        pbx = np.zeros((nB,nX))
        g0 = 1.0/2.0
        g1 = 1.0/2.0
        g2 = 1.0 - g0 - g1

        # Events: 
        # b=0 -> ... | ...
        # b=1 ->  *  | ...
        # b=2 -> ... |  *
        # b=3 ->  *  |  *

        pbx[0][0] = (1.0-dc)*((1.0-dc)*(1.0-olap) + dc)   
        pbx[1][0] = (1.0-dc)*dc*olap   
        pbx[2][0] = (1.0-dc)*(1.0-dc)*olap 
        double_click_0 = 1.0-pbx[0][0]-pbx[1][0]-pbx[2][0]   

        pbx[0][1] = (1.0-dc)*dc*olap            
        pbx[1][1] = (1.0-dc)*((1.0-dc)*(1.0-olap) + dc)   
        pbx[2][1] = (1.0-dc)*(1.0-dc)*olap  
        double_click_1 = 1.0-pbx[0][1]-pbx[1][1]-pbx[2][1]

        pbx[0][2] = (1.0-dc)*(1.0-dc)*(1.0-np.exp(-np.abs(beta0)**2.0))*np.exp(-np.abs(beta1)**2.0) + dc*(1.0-dc) 
        pbx[1][2] = (1.0-dc)*(1.0-dc)*(1.0-np.exp(-np.abs(beta1)**2.0))*np.exp(-np.abs(beta0)**2.0) + dc*(1.0-dc) 
        pbx[2][2] = (1.0-dc)*(1.0-dc)*np.exp(-np.abs(beta0)**2.0)*np.exp(-np.abs(beta1)**2.0) + dc*dc            
        double_click_2 =  1.0-pbx[0][2]-pbx[1][2]-pbx[2][2]

        #Distribute_probabilities
        pbx[0][0] = pbx[0][0] + g0*double_click_0
        pbx[1][0] = pbx[1][0] + g1*double_click_0
        pbx[2][0] = pbx[2][0] + g2*double_click_0

        pbx[0][1] = pbx[0][1] + g0*double_click_1
        pbx[1][1] = pbx[1][1] + g1*double_click_1
        pbx[2][1] = pbx[2][1] + g2*double_click_1

        pbx[0][2] = pbx[0][2] + g0*double_click_2
        pbx[1][2] = pbx[1][2] + g1*double_click_2
        pbx[2][2] = pbx[2][2] + g2*double_click_2
                
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

        #Run the SDPs
        start = time.process_time()
        out_Hmin = Hmin(rho,pbx,nX,nB,3,xstar)
        out_H = H(m-1,w,t,rho,pbx,nX,nB,3,xstar)
        end = time.process_time()
        
        H_arr[i][j] = out_H
        Hmin_arr[i][j] = out_Hmin
        
        print('RESULT')
        print(out_H)
        print(out_Hmin)
        print('in',end-start,'seconds')
        
        