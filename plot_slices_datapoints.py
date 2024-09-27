#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 09:31:09 2024

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


# ---------------------------------------------------------------------------- 
# ----------------------------------------------------------------------------
# PLOTTING STATION
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

fig, ax = plt.subplots(2,1,figsize=(4,5),sharex=False, sharey=False)

N = 100
a_vec = np.linspace(0.2,0.6,N) #Vector of alphas
b_vec = np.linspace(0.6,0.7,N) #Vector of betas

target_alpha = sigle_datapoint_stored[0][0]
target_beta = sigle_datapoint_stored[1][0]

#ax[0].axvline(target_alpha,ls='--',c='black',lw=1)
#ax[1].axvline(target_beta,ls='--',c='black',lw=1)

x_label = [r'$\alpha$',r'$\beta$']
x_axis_rotation = [0,30]
x_vec = [a_vec,b_vec]
lab_alpha = 0.401
lab_beta = 0.641
amp_t = [lab_alpha,lab_beta]
amp_err = [0.007,0.006]

Hmin_datapoint = sigle_datapoint_stored[2]#1.1083317034176077
AEP_datapoint = sigle_datapoint_stored[4]#1.3182740010072045
EAT_datapoint = sigle_datapoint_stored[6]#1.3130914551967359

#ax[0].plot(H_slice_vs_alpha[0],H_slice_vs_alpha[1])
#ax[1].plot(H_slice_vs_beta[0],H_slice_vs_beta[1])

xvec = H_slice_vs_alpha[0]
func = []
p0vec = []
p1vec = []
p2vec = []
for xx in xvec:
    p0vec += [ 0.25*(1.0-np.exp(-(xx**2.0))) + 0.5*(1.0-np.exp(-target_beta**2.0))*np.exp(-target_beta**2.0) ]
    p1vec += [ 0.5*(np.exp(-(xx**2.0)) + np.exp(-2.0*target_beta**2.0)) ]
    p2vec += [ 0.5*(np.exp(-(xx**2.0)) + np.exp(-2.0*target_beta**2.0)) ]
    
    func += [ xx ]
#ax[0].plot(p2vec,H_slice_vs_alpha[1],c='purple',ls='--')

Hmin_vec_slices = [ Hmin_vec_slices_0 , Hmin_vec_slices_1 ]
H_vec_slices = [ H_vec_slices_0 , H_vec_slices_1 ]

for s in range(2):
    
    ax[s].tick_params(axis='both', direction='in', right=True, top=True, width=1)
    ax[s].set_ylabel(r'Randomness', size=15,rotation=90,labelpad=5)
    ax[s].set_yticks([0.9,1.0,1.1,1.2,1.3,1.4])
    ax[s].tick_params(axis='x', rotation=x_axis_rotation[s])
    ax[s].set_xlabel(x_label[s], size=15,labelpad=-2)
    ax[s].plot(Hmin_vec_slices[s][0],Hmin_vec_slices[s][1],c='red',ls='-',label=r'$H_{min}$',lw=1)
    ax[s].plot(H_vec_slices[s][0],H_vec_slices[s][1],c='black',lw=1,ls='-',label='S')
    
    #ax[s].plot(H_GEAT[s][0],H_GEAT[s][1],c='blue',lw=1,alpha=1.0,ls='-',label='non i.i.d.')
    #ax[s].plot(H_QAEP[s][0],H_QAEP[s][1],c='magenta',lw=1,alpha=1.0,ls='--',label='i.i.d.')
    
    ax[s].axhline(1.0,c="gray",ls='--')
    
    ax[s].scatter(amp_t[s],Hmin_datapoint[0],c='none',marker='s',lw=1,edgecolors='r')
    ax[s].errorbar(amp_t[s],Hmin_datapoint[0],yerr=Hmin_datapoint[1],xerr=amp_err[s],c='red',capsize=2,lw=1)

    ax[s].errorbar(amp_t[s],AEP_datapoint[0],yerr=AEP_datapoint[1],xerr=amp_err[s],c='magenta',capsize=2,lw=1)
    ax[s].scatter(amp_t[s],AEP_datapoint[0],c='magenta',marker='x')#,label='i.i.d.')

    ax[s].errorbar(amp_t[s],EAT_datapoint[0],yerr=EAT_datapoint[1],xerr=amp_err[s],c='blue',capsize=2,lw=1)
    ax[s].scatter(amp_t[s],EAT_datapoint[0],c='blue',marker='.')#,label='non i.i.d.')

ax[0].set_ylim(0.89,1.4) 
ax[0].set_xlim(0.2,0.6) 

ax[1].set_ylim(0.89,1.4) 
ax[1].set_xlim(0.6,0.7) 

ax[0].scatter(0.201,1.485,clip_on=False,c='none',marker='s',lw=1,edgecolors='r')
ax[0].scatter(0.47,1.49,clip_on=False,c='blue',marker='.')
ax[0].scatter(0.47,1.445,clip_on=False,c='magenta',marker='x')

ax[0].text(0.49,1.475,'non i.i.d.')
ax[0].text(0.49,1.43,'i.i.d.')
#-----

ax[0].text(0.1,1.45,'b)',size=16)

plt.subplots_adjust(left=0.2,
                    bottom=0.1, 
                    right=0.8, 
                    top=0.88, 
                    wspace=0.08, 
                    hspace=0.2)

ax[0].legend(bbox_to_anchor=(0.6, 1.23),fontsize=10,frameon=False,ncols=3,loc='upper right')

#plt.savefig('Outputs/entropy_datapoints.pdf')

plt.show()