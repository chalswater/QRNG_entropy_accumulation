#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 13:29:13 2024

@author: carles
"""

#np.savetxt('Outputs/H_fmin.csv', H_datapoint_vec, delimiter =", ", fmt ='% s') 
H_fmin = np.genfromtxt('Outputs/H_fmin.csv', delimiter=",")

fig, ax = plt.subplots(1,3,figsize=(7,2),sharex=False, sharey=True)

ee = np.sqrt(np.var(H_fmin))

ax[0].plot(p_vec_b0x2,H_fmin_saved,ls='',marker='.',c='black')
#ax[0].errorbar(p_vec[0][2],H_fmin,yerr=ee,capsize=2,c='black',lw=1,ls='None')
ax[1].plot(p_vec_b1x2,H_fmin_saved,ls='',marker='.',c='black')
#ax[1].errorbar(p_vec[1][2],H_fmin,yerr=ee,capsize=2,c='black',lw=1,ls='None')
ax[2].plot(p_vec_b2x2,H_fmin_saved,ls='',marker='.',c='black')
#ax[2].errorbar(p_vec[2][2],H_fmin,yerr=ee,capsize=2,c='black',lw=1,ls='None')

ax[0].axhline(np.min(H_fmin)-ee-1e-3,ls='-',c='red')
ax[1].axhline(np.min(H_fmin)-ee-1e-3,ls='-',c='red')
ax[2].axhline(np.min(H_fmin)-ee-1e-3,ls='-',c='red')

print(np.min(H_fmin)-ee-1e-3)
print(ee)

ax[0].set_xlim(0.26,0.275)
ax[0].set_xticks([0.26,0.27])
ax[0].set_xlabel(r'$p(0|2)$')

ax[0].set_ylim(1.319,1.332)
ax[0].set_yticks([1.320,1.322,1.324,1.326,1.328,1.330,1.332])
ax[0].set_ylabel(r'Shannon entropy, $H^\ast$')

ax[1].set_xlim(0.258,0.272)
ax[1].set_xticks([0.26,0.27])
ax[1].set_xlabel(r'$p(1|2)$')

ax[2].set_xlim(0.454,0.482)
ax[2].set_xticks([0.46,0.47,0.48])
ax[2].set_xlabel(r'$p(2|2)$')

fig.tight_layout()

#plt.savefig('Outputs/fmin_figure.pdf')