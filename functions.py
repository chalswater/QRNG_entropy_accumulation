#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 16:47:41 2024

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

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------

def half(lista):
    out = np.array([lista[i] for i in range(len(lista)) if (i%3)==0])
    return out[1:len(out)]

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------


def chunkify(lst,n):
    return [lst[i::n] for i in range(n)]

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------

def P_click(click,a):
    if click == 1:
        return (1.0-np.exp(-np.abs(a)**2.0))
    elif click == 0:
        return np.exp(-np.abs(a)**2.0)
    else:
        return 'ERROR: Enter a valid click value (0,1)'
    
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
    
def deltaF(x,n):
    if x == n:
        return 1.0
    else:
        return 0.0
    
    
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------

def Hmin(rho,pbx,nX,nB,dim,xstar):
    
    """ Min-entropy with a known state preparation """
    #xstar = 0 # Get randomness from a specific state preparation
    
    # --------------
    # Variables
    # --------------
    
    M = {}
    for l in range(nB):
        M[l] = {}
        for b in range(nB):
            M[l][b] = cp.Variable((dim,dim),complex = True)
            
    # --------------
    # Constraints
    # --------------
        
    ct = []
    for l in range(nB):
        suma = 0.0
        for b in range(nB):
            ct += [M[l][b] >> 0.0]
            suma = suma + M[l][b]
        ct += [suma == 1.0/float(dim)*cp.trace(suma)*np.identity(dim)]
        
    for x in range(nX):
        for b in range(nB):
            suma = 0.0
            for l in range(nB):
                suma = suma + cp.real(cp.trace(rho[x] @ M[l][b]))
            ct += [pbx[b][x] == suma]
            
    # --------------
    # Object function: Guessing probability
    # --------------
            
    pg = 0.0
    for l in range(nB):
        pg = pg + cp.real(cp.trace(rho[xstar] @ M[l][l]))
        
    # --------------
    # Run the SDP
    # --------------
            
    obj = cp.Maximize(pg)
    prob = cp.Problem(obj,ct)

    try:
        mosek_params = {
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1
            }
        prob.solve(solver='MOSEK',verbose=False, mosek_params=mosek_params)

    except SolverError:
        something = 10
        
    # --------------
    # Output
    # --------------
    
    if pg.value != None:
        return -np.log2(pg.value)
    else:
        return None
    
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------

    
def H_old(m,w,t,rho,pbx,nX,nB,dim,xstar):
    
    """ Shannon entropy with a known state preparation """
    #xstar = 0 # Get randomness from a specific state preparation
    
    # --------------
    # Variables
    # --------------
    
    pi = {}
    for b in range(nB):
        pi[b] = cp.Variable((dim,dim),complex=True)

    xi = {}
    for i in range(m):
        xi[i] = {}
        for b in range(nB):
            xi[i][b] = {}
            for bb in range(nB):
                xi[i][b][bb] = cp.Variable((dim,dim),complex=True)

    eta = {}
    for i in range(m):
        eta[i] = {}
        for b in range(nB):
            eta[i][b] = {}
            for bb in range(nB):
                eta[i][b][bb] = cp.Variable((dim,dim),complex=True)

    # --------------
    # Constraints
    # --------------
                
    ct = []

    for i in range(m):
        for bb in range(nB):
            suma_xi = 0.0
            suma_eta = 0.0
            for b in range(nB):
                suma_xi = suma_xi + xi[i][b][bb]
                suma_eta = suma_eta + eta[i][b][bb]

                G = cp.bmat([[ pi[b]         ,  xi[i][b][bb] ],
                             [ xi[i][b][bb] , eta[i][b][bb] ]])
                ct += [G >> 0.0]

            ct += [suma_xi == cp.trace(suma_xi)*np.identity(dim)/float(dim)]
            ct += [suma_eta == cp.trace(suma_eta)*np.identity(dim)/float(dim)]       

    suma = 0.0
    for b in range(nB):
        suma = suma + pi[b]
    ct += [suma == np.identity(dim)]
    
    for b in range(nB):
        for x in range(nX):
            ct += [ cp.trace(pi[b]@rho[x]) == pbx[b][x]]

    # --------------
    # Object function: Shannon entropy
    # --------------
                          
    H = 0.0
    for i in range(m):
        for b in range(nB):
            suma = 0.0
            for bb in range(nB):
                suma = suma + eta[i][bb][b]
            H += w[i]/(t[i]*np.log(2.0)) * cp.real(cp.trace(rho[xstar] @ (2.0*xi[i][b][b] + \
                                                                   (1.0-t[i])*eta[i][b][b] + \
                                                                         t[i]*suma ) ) )

    cm = 0.0
    for i in range(m):
        cm += w[i]/(t[i]*np.log(2.0))

    H += cm

    # --------------
    # Run the SDP
    # --------------
    
    obj = cp.Minimize(H)
    prob = cp.Problem(obj,ct)

    output = []

    try:
        mosek_params = {
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1
            }
        prob.solve(solver='MOSEK',verbose=False, mosek_params=mosek_params)

    except SolverError:
        something = 10
        
    # --------------
    # Output
    # --------------
        
    return H.value
    
    
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------

def H(m,w,t,rho,pbx,nX,nB,dim,xstar):
    
    """ Shannon entropy with a known state preparation """
    #xstar = 0 # Get randomness from a specific state preparation
            
    H_out = sum([ w[i]/(t[i]*np.log(2.0)) for i in range(m) ])
    
    for i in range(m):
        
        # --------------
        # Variables
        # --------------
        
        pi = {}
        for b in range(nB):
            pi[b] = cp.Variable((dim,dim),complex=True)
    
        xi = {}
        for b in range(nB):
            xi[b] = {}
            for bb in range(nB):
                xi[b][bb] = cp.Variable((dim,dim),complex=True)
    
        eta = {}
        for b in range(nB):
            eta[b] = {}
            for bb in range(nB):
                eta[b][bb] = cp.Variable((dim,dim),complex=True)
    
        # --------------
        # Constraints
        # --------------
                    
        ct = []

        for bb in range(nB):
            suma_xi = 0.0
            suma_eta = 0.0
            for b in range(nB):
                suma_xi = suma_xi + xi[b][bb]
                suma_eta = suma_eta + eta[b][bb]

                G = cp.bmat([[ pi[b]     ,  xi[b][bb] ],
                             [ xi[b][bb] , eta[b][bb] ]])
                ct += [G >> 0.0]

            ct += [suma_xi == cp.trace(suma_xi)*np.identity(dim)/float(dim)]
            ct += [suma_eta == cp.trace(suma_eta)*np.identity(dim)/float(dim)]       
    
        suma = 0.0
        for b in range(nB):
            suma = suma + pi[b]
        ct += [suma == np.identity(dim)]
        
        for b in range(nB):
            for x in range(nX):
                ct += [ cp.trace(pi[b]@rho[x]) == pbx[b][x]]
    
        # --------------
        # Object function: Shannon entropy
        # --------------
                              
        H = 0.0
        for b in range(nB):
            suma = 0.0
            for bb in range(nB):
                suma = suma + eta[bb][b]
            H += w[i]/(t[i]*np.log(2.0)) * cp.real(cp.trace(rho[xstar] @ (2.0*xi[b][b] + \
                                                                   (1.0-t[i])*eta[b][b] + \
                                                                         t[i]*suma ) ) )
    
        # --------------
        # Run the SDP
        # --------------
        
        obj = cp.Minimize(H)
        prob = cp.Problem(obj,ct)
    
        output = []
    
        try:
            mosek_params = {
                    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1
                }
            prob.solve(solver='MOSEK',verbose=False, mosek_params=mosek_params)
    
        except SolverError:
            something = 10
            
        if H.value != None:
            H_out += H.value
        else:
            H_out = None
            break
        
    # --------------
    # Output
    # --------------
        
    return H_out

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------

def rng_scaling(ns,data):
    
    """ 
    For a fixed ampunt of data 'ns' and given 'data' compute the following:
 
        - Target amplitudes (before noise) alpha, beta and their standard deviations
        - Shannon and min-entropies with their respective standard deviations
        - Certifiable randomness according to AEP and EAT
        
    Data from experiment should come in the form:
    
        data = []
        with h5py.File('SDI_QRNG.hdf5', 'r') as f:
            for run in range(11):
                data += [np.array(f[f'/events/set{run+1}'])]
                print('\r'+f'Run:{run}\r',end="")
                
        
    """
    
    nBB = 4 #True number of outcomes
    nB = 3 #Number of outcomes (when re-labling outcome b=3)
    nX = 3 #Number of states
    px = [0.25,0.25,0.5]
    
    epsilon = 1e-8 #EAT epsilon
    
    n_resamplings = 11 # Number of resamplings we need to do
    n_bins = int(1e7/ns)

    xstar = 2 # From which setting we extract randomness
    
    alpha_vec = []
    beta_vec = []
    dc_vec = []
    
    Hmin_datapoint_vec = []
    H_datapoint_vec = []
    AEP_datapoint_vec = []
    EAT_datapoint_vec = []
    
    remaining_time = 0.0
    
    # Gauss-Radau quadrature
    m_in = 4
    m = int(m_in*2)
    distribution = chaospy.Uniform(lower=1e-3, upper=1)
    t, w = chaospy.quadrature.radau(m_in,distribution,1.0)
    t = t[0]
    
    for nn in range(n_resamplings):
        
        lista = chunkify(data[nn], n_bins)
        
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
    
    # Generalised Entropy Accumulation Theorem
    #f_tradeoff = np.min(H_datapoint_vec) - np.std(H_datapoint_vec) - 1e-3
    f_tradeoff = H_datapoint_vec - np.std(H_datapoint_vec) - 1e-3
    Var = np.var(f_tradeoff)
    Min = np.min(f_tradeoff)
    Max = np.max(f_tradeoff)
    
    for nn in range(n_resamplings):
        
        p_omega = 0.5
        aa = 1.0 + 1.0/np.sqrt(ns)
        
        g_eps = -np.log(1.0-np.sqrt(1.0-epsilon**2.0))
        exponent = (2.0*np.log(nB)+Max-Min)
        inside = 2.0**exponent+np.exp(2.0)
        Kaa = (2.0-aa)**3.0/(6.0*(3.0-2.0*aa)**3.0*np.log(2.0))*2.0**((aa-1.0)/(2.0-aa)*exponent)*(np.log(inside))**3.0
        V = np.log(2.0*nB**2.0+1.0) + np.sqrt(2.0+Var)
        
        EAT_datapoint_vec += [f_tradeoff[nn] - (aa-1.0)/(2.0-aa)*np.log(2.0)/2.0*V**2.0 - \
               ((g_eps + aa*np.log(1.0/p_omega))/(aa-1.0))/ns - \
               ((aa-1.0)/(2.0-aa))**2.0 * Kaa]
        
    output = []
            
    target_alpha = sum([ alpha_vec[i]/len(alpha_vec) for i in range(len(alpha_vec))])
    target_alpha_var = np.std(alpha_vec)
    output += [[target_alpha,target_alpha_var]] # 0
    
    print('alpha')
    print(target_alpha,target_alpha_var)
    
    target_beta = sum([ beta_vec[i]/len(beta_vec) for i in range(len(beta_vec))])
    target_beta_var = np.std(beta_vec)
    output += [[target_beta,target_beta_var]] # 1
    
    print('beta')
    print(target_beta,target_beta_var)
            
    Hmin_avg = sum([ Hmin_datapoint_vec[i]/len(Hmin_datapoint_vec) for i in range(len(Hmin_datapoint_vec))])
    Hmin_var = np.std(Hmin_datapoint_vec)
    output += [[Hmin_avg,Hmin_var]] # 2
    
    print('Hmin')
    print(Hmin_avg,Hmin_var)
    
    H_avg = sum([ H_datapoint_vec[i]/len(H_datapoint_vec) for i in range(len(H_datapoint_vec))])
    H_var = np.std(H_datapoint_vec)
    output += [[H_avg,H_var]] # 3
    
    print('Shannon entropy')
    print(H_avg,H_var)
    
    AEP_avg = sum([ AEP_datapoint_vec[i]/len(AEP_datapoint_vec) for i in range(len(AEP_datapoint_vec))])
    AEP_var = np.std(AEP_datapoint_vec)
    output += [[AEP_avg,AEP_var]] # 4
    
    print('AEP randomness')
    print(AEP_avg,AEP_var)
    
    dark_count_avg = sum([ dc_vec[i]/len(dc_vec) for i in range(len(dc_vec))])
    dark_count_var = np.std(dc_vec)
    output += [[dark_count_avg,dark_count_var]] # 5
    
    EAT_avg = sum([ EAT_datapoint_vec[i]/len(EAT_datapoint_vec) for i in range(len(EAT_datapoint_vec))])
    EAT_var = np.std(EAT_datapoint_vec)
    output += [[EAT_avg,EAT_var]] # 6
    
    print('EAT randomness')
    print(EAT_avg,EAT_var)
    
    return output













