#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:27:53 2024

@author: carles
"""

import matplotlib.pyplot as plt

N = 30
ns_vec = np.linspace(3.0,7.0,N)

print(ns_vec[N-2:N])
fig, ax = plt.subplots(1,1,figsize=(4,3),sharex=False, sharey=False)

ax.plot(10.0**ns_vec,QAEP_scalability,ls='-',c='magenta',lw=0.5,marker='None')
ax.plot(10.0**half(ns_vec),half(GEAT_scalability),ls='-',c='blue',lw=0.5,marker='none')
ax.plot(10.0**ns_vec[N-3:N],GEAT_scalability[N-3:N],ls='-',c='blue',lw=0.5,marker='none')

ax.scatter(10.0**half(ns_vec),half(QAEP_scalability),ls='None',c='magenta',lw=1,marker='x',edgecolors='magenta')
ax.scatter(10.0**half(ns_vec),half(GEAT_scalability),ls='None',c='blue',lw=1,marker='.',edgecolors='b')
ax.scatter(10.0**half(ns_vec),half(Hmin_scalability),ls='None',c='none',lw=1,marker='s',edgecolors='r')

ax.errorbar(10.0**half(ns_vec),half(QAEP_scalability),yerr=half(QAEP_scalability_var),capsize=2,c='magenta',lw=1,ls='None')
ax.errorbar(10.0**half(ns_vec),half(GEAT_scalability),yerr=half(GEAT_scalability_var),capsize=2,c='blue',lw=1,ls='None')
ax.errorbar(10.0**half(ns_vec),half(Hmin_scalability),yerr=half(Hmin_scalability_var),capsize=2,c='red',lw=1,ls='None')

ax.axhline(1.32,c='black',lw=1)
ax.axhline(Hmin_scalability[-1],c='red',lw=1)

ax.set_xscale('symlog')
ax.set_ylim(0.6,1.41)
ax.set_xlim(10.0**3.0,10.0**7.0)
ax.tick_params(axis='both', direction='in', right=True, top=True, width=1)

ax.text(11**3,0.8,'i.i.d. (AEP)',rotation=58,c='magenta',size=13)
ax.text(11**3.7,0.67,'non i.i.d. (EAT)',rotation=61,c='blue',size=13)

ax.text(10.0**3.1,1.345,r'Shannon entropy at $\alpha_{T}$ and $\beta_{T}$',c='black',rotation=0,size=12)
ax.text(10.0**5.6,1.02,r'Min-entropy',c='red',rotation=0,size=12)
ax.text(10.0**5.6,0.96,r'at $\alpha_{T}$ and $\beta_{T}$',c='red',rotation=0,size=12)



ax.set_xlabel('Total number of samples',size=13)
ax.set_ylabel('Randomness',size=13)

plt.subplots_adjust(left=0.14,
                    bottom=0.16, 
                    right=0.97, 
                    top=0.97, 
                    wspace=0.08, 
                    hspace=0.2)

#plt.savefig('Outputs/n_rounds_scalability_new.pdf')

