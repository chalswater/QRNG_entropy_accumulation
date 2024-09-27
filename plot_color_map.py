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

from matplotlib import cm
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

n_blues = 400
n_reds = 200

top = cm.get_cmap('Blues_r', n_blues)
bottom = cm.get_cmap('Reds', n_reds)

newcolors = np.vstack((top(np.linspace(0, 1, n_blues)),
                       bottom(np.linspace(0, 1, n_reds))))
newcmp = ListedColormap(newcolors, name='RedBlues7')

""" IMPORTANT """
# Comment this line if it's the second time you compile this file !!
mpl.pyplot.register_cmap(cmap=newcmp)

a_vec = np.linspace(0.0,1.0,100) #Vector of alphas
b_vec = np.linspace(0.3,0.9,100) #Vector of betas

#fig = plt.figure(figsize=(4,5),constrained_layout=True)
#ax = fig.add_gridspec(2,1,hspace=-1.0,wspace=-1.0)
fig, ax = plt.subplots(2,1,figsize=(4,5),sharex=False, sharey=False)

#ax[0] = fig.add_subplot(ax[0, 0],sharex = ax1)
ax[0].tick_params(axis='both', direction='in', right=True, top=True, width=1)
ax[0].tick_params(axis='x', rotation=30)
ax[0].tick_params('x', labelbottom=False)
ax[0].set_ylim(0.01,1.0)
ax[0].set_xlim(0.3,0.9)
ax[0].set_ylabel(r'$\alpha$',fontsize=15,rotation=0,labelpad=5)
#ax[0].set_yticks([0.25,0.35,0.45,0.55])

#ax[1] = fig.add_subplot(ax[1, 0],sharex = ax2)
ax[1].tick_params(axis='both', direction='in', right=True, top=True, width=1)
ax[1].tick_params(axis='x', rotation=30)
ax[1].set_xlabel(r'$\beta$',fontsize=15,labelpad=-2)
ax[0].set_ylim(0.01,1.0)
ax[0].set_xlim(0.3,0.9)
ax[1].set_ylabel(r'$\alpha$',fontsize=15,rotation=0,labelpad=5)


X, Y = np.meshgrid(b_vec, a_vec)

target_alpha_teo = 0.4
target_beta_teo = 0.66

cf1 = ax[0].pcolor(X,Y,Hmin_array_is, vmin=0.0, vmax=1.5,shading='auto',cmap = 'RedBlues7')
cf2 = ax[1].pcolor(X,Y,H_array_is, vmin=0.0, vmax=1.5,shading='auto',cmap = 'RedBlues7')#,cmap='myColorMap')#cmap = 'RdBu_r')

ax[0].scatter(target_beta_teo, target_alpha_teo, s=80, facecolors='none', edgecolors='black',lw=1)
ax[0].scatter(target_beta_teo, target_alpha_teo, marker='+', c='black',lw=1)

ax[1].scatter(target_beta_teo, target_alpha_teo, s=80, facecolors='none', edgecolors='black',lw=1)
ax[1].scatter(target_beta_teo, target_alpha_teo, marker='+', c='black',lw=1)

ax[0].axvline(target_beta_teo,ls='--',c='black',lw=1)
ax[0].axhline(target_alpha_teo,ls='--',c='black',lw=1)

ax[1].axvline(target_beta_teo,ls='--',c='black',lw=1)
ax[1].axhline(target_alpha_teo,ls='--',c='black',lw=1)

#-----

cbar_ax1 = fig.add_axes([0.85, 0.53 , 0.07, 0.3])
cbar_ax1.set_title(r'$H_{min}$', size=14)
fig.colorbar(cf1, cax=cbar_ax1,orientation='vertical')

cbar_ax2 = fig.add_axes([0.85, 0.13 , 0.07, 0.3])
cbar_ax2.set_title(r'$H$', size=14)
fig.colorbar(cf2, cax=cbar_ax2,orientation='vertical')

#-----

ax[0].text(0.32,0.85,'Min-entropy $H_{min}$',size=12)
ax[1].text(0.32,0.85,'Shannon',size=12)
ax[1].text(0.32,0.75,'entropy $H$',size=12)

ax[0].text(0.3,1.1,'a)',size=16)

plt.subplots_adjust(left=0.101,
                    bottom=0.1, 
                    right=0.8, 
                    top=0.88, 
                    wspace=0.08, 
                    hspace=0.1)

ax[0].set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
ax[1].set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
ax[1].set_xticks([0.3,0.4,0.5,0.6,0.7,0.8,0.9])

#plt.savefig('Outputs/H_Hmin_compare.pdf')

plt.show()