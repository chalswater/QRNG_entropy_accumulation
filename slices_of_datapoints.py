#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 09:24:44 2024

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

H_vec = [[[],[]],[[],[]]] #Here we collect H
Hmin_vec = [[[],[]],[[],[]]]
H_QAEP = [[[],[]],[[],[]]]
H_GEAT = [[[],[]],[[],[]]]

N = 100
a_vec = np.linspace(0.2,0.6,N) #Vector of alphas
b_vec = np.linspace(0.6,0.7,N) #Vector of betas

nBB = 4 #True number of outcomes
nB = 3 #Number of outcomes (when re-labling outcome b=3)
nX = 3 #Number of states
nS = 2 #Number of scenarios
xstar = 2 #Which state we get randomness from

epsilon = 1e-8 #EAT epsilon
epsilon_fs = 1e-8 #Finite-size effects epsilon
dc = 4.485615570543283e-06

id_2 = np.identity(2)
sigmax = np.array([[0.0,1.0],[1.0,0.0]])
sigmay = np.array([[0.0,-1.0j],[1.0j,0.0]])
sigmaz = np.array([[1.0,0.0],[0.0,-1.0]])

px = [0.25,0.25,0.5]

# Gauss-Radau quadrature
m_in = 4
m = int(m_in*2)
distribution = chaospy.Uniform(lower=1e-3, upper=1)
t, w = chaospy.quadrature.radau(m_in,distribution,1.0)
t = t[0]

eta = 0.06

lab_alpha = 0.401
lab_beta = 0.641

for s in range(2):
    for i in range(N):
             
        if s == 0:
            alpha = a_vec[i]#target_alpha#a_vec[i]#0.726#b_vec[i]#0.74#0.42#a_vec[i]#0.82#a_vec[i]#0.78
            x_var = alpha
            beta0 = lab_beta#b_vec[j]
            beta1 = lab_beta#b_vec[j]
        elif s == 1:
            alpha = lab_alpha#a_vec[i]#0.726#b_vec[i]#0.74#0.42#a_vec[i]#0.82#a_vec[i]#0.78
            beta0 = b_vec[i]
            beta1 = b_vec[i]
            x_var = beta0
            
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
    
        start = time.process_time()
        #Run the SDPs (either the Primal or the Dual). The dual is used to apply fs and EAT corrections
        out_Hmin = Hmin(rho,pbx,nX,nB,3,xstar)
        out_H = H(m-1,w,t,rho,pbx,nX,nB,3,xstar)
        end = time.process_time() 
        
        Hmin_value = out_Hmin
        H_value = out_H
        
        # Assymptotic Equipartiotion Property (non iid)
        #pb = [sum([px[x]*pbx[b][x] for x in range(nX)]) for b in range(nB)]
        #Hmax = 2.0*np.log(sum([np.sqrt(pb[b]) for b in range(nB)]))
        #etaAEP = np.sqrt(2.0**(-Hmin_value)) + np.sqrt(2.0**Hmax) + 1.0
        #delta = 4.0*np.log(etaAEP)*np.sqrt(np.log(2.0/epsilon**2.0))
        
        #H_QAEP[s][0] += [x_var]
        #H_QAEP[s][1] += [H_value - delta/np.sqrt(ns)]
    
        Hmin_vec[s][0] += [x_var]
        Hmin_vec[s][1] += [out_Hmin]
        H_vec[s][0] += [x_var]
        H_vec[s][1] += [out_H]
        
        #H_GEAT[s][0] += [x_var]

        print('RESULT')
        print(out_H)
        print(out_Hmin)
        #print(H_QAEP[s][1][i])
        print('in',end-start,'seconds')
        
        
    # Generalised Entropy Accumulation Theorem
    #f_tradeoff = H_vec[s][1]
    #Var = np.var(f_tradeoff)
    #Min = np.min(f_tradeoff)
    #Max = np.max(f_tradeoff)
    #pg_vec = [2.0**(-Hmin_vec[s][i]) for i in range(N)]
    
    #for i in range(N):
        
    #    p_omega = 0.5
    #    aa = 1.0 + 1.0/np.sqrt(ns)
        
    #    g_eps = -np.log(1.0-np.sqrt(1.0-epsilon**2.0))
    #    exponent = (2.0*np.log(nB)+Max-Min)
    #    inside = 2.0**exponent+np.exp(2.0)
    #    Kaa = (2.0-aa)**3.0/(6.0*(3.0-2.0*aa)**3.0*np.log(2.0))*2.0**((aa-1.0)/(2.0-aa)*exponent)*(np.log(inside))**3.0
    #    V = np.log(2.0*nB**2.0+1.0) + np.sqrt(2.0+Var)
        
    #    H_GEAT[s][1] += [f_tradeoff[i] - (aa-1.0)/(2.0-aa)*np.log(2.0)/2.0*V**2.0 - \
    #           ((g_eps + aa*np.log(1.0/p_omega))/(aa-1.0))/ns - \
    #           ((aa-1.0)/(2.0-aa))**2.0 * Kaa]
        
