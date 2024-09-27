#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 09:23:25 2024

@author: carles
"""

import numpy as np    

# Read and write cvx files

#np.savetxt('Outputs/Hmin_array_100_is.csv', Hmin_arr, delimiter =", ", fmt ='% s') 
#np.savetxt('Outputs/H_array_100_is.csv', H_arr, delimiter =", ", fmt ='% s') 

#np.savetxt('Outputs/H_slice_vs_beta.csv', H_vec[1], delimiter =", ", fmt ='% s') 

#np.savetxt('Outputs/AEP_datapoint_vec_ns1e7.csv', AEP_datapoint_vec, delimiter =", ", fmt ='% s') 
#np.savetxt('Outputs/EAT_datapoint_vec_ns1e7.csv', EAT_datapoint_vec, delimiter =", ", fmt ='% s') 

#np.savetxt('Outputs/H_vec_nrounds.csv', H_vec_nrounds, delimiter =", ", fmt ='% s') 
#np.savetxt('Outputs/Hmin_vec_nrounds.csv', Hmin_vec_nrounds, delimiter =", ", fmt ='% s') 
#np.savetxt('Outputs/H_QAEP_nrounds.csv', H_QAEP_nrounds, delimiter =", ", fmt ='% s') 
#np.savetxt('Outputs/H_GEAT_nrounds.csv', H_GEAT_nrounds, delimiter =", ", fmt ='% s') 

#np.savetxt('Outputs/H_vec_nrounds_var.csv', H_vec_nrounds_var , delimiter =", ", fmt ='% s') 
#np.savetxt('Outputs/Hmin_vec_nrounds_var.csv', Hmin_vec_nrounds_var , delimiter =", ", fmt ='% s') 
#np.savetxt('Outputs/H_QAEP_nrounds_var.csv', H_QAEP_nrounds_var , delimiter =", ", fmt ='% s') 
#np.savetxt('Outputs/H_GEAT_nrounds_var.csv', H_GEAT_nrounds_var , delimiter =", ", fmt ='% s') 

#np.savetxt('Outputs/sigle_datapoint.csv', sigle_datapoint , delimiter =", ", fmt ='% s') 

#np.savetxt('Outputs/Hmin_vec_slices_0.csv', Hmin_vec[0] , delimiter =", ", fmt ='% s') 
#np.savetxt('Outputs/Hmin_vec_slices_1.csv', Hmin_vec[1] , delimiter =", ", fmt ='% s') 
#np.savetxt('Outputs/H_vec_slices_0.csv', H_vec[0] , delimiter =", ", fmt ='% s') 
#np.savetxt('Outputs/H_vec_slices_1.csv', H_vec[1] , delimiter =", ", fmt ='% s') 

#np.savetxt('Outputs/p_vec_b0x2.csv', p_vec[0][2] , delimiter =", ", fmt ='% s') 
#np.savetxt('Outputs/p_vec_b1x2.csv', p_vec[1][2] , delimiter =", ", fmt ='% s') 
#np.savetxt('Outputs/p_vec_b2x2.csv', p_vec[2][2] , delimiter =", ", fmt ='% s') 

#np.savetxt('Outputs/H_fmin_saved.csv', H_fmin , delimiter =", ", fmt ='% s') 

# Min-tradeoff function data
p_vec_b0x2 = np.genfromtxt('Outputs/p_vec_b0x2.csv', delimiter=",")
p_vec_b1x2 = np.genfromtxt('Outputs/p_vec_b1x2.csv', delimiter=",")
p_vec_b2x2 = np.genfromtxt('Outputs/p_vec_b2x2.csv', delimiter=",")

H_fmin_saved = np.genfromtxt('Outputs/H_fmin_saved.csv', delimiter=",")

# Hmin and Shannon entropies for slices of datapoints
Hmin_vec_slices_0 = np.genfromtxt('Outputs/Hmin_vec_slices_0.csv', delimiter=",")
Hmin_vec_slices_1 = np.genfromtxt('Outputs/Hmin_vec_slices_1.csv', delimiter=",")
H_vec_slices_0 = np.genfromtxt('Outputs/H_vec_slices_0.csv', delimiter=",")
H_vec_slices_1 = np.genfromtxt('Outputs/H_vec_slices_1.csv', delimiter=",")

# Single datapoint from the slices
sigle_datapoint_stored = np.genfromtxt('Outputs/sigle_datapoint.csv', delimiter=",")
print(sigle_datapoint_stored)
# Scalability of our method
Hmin_scalability = np.genfromtxt('Outputs/Hmin_vec_nrounds.csv', delimiter=",")
H_scalability = np.genfromtxt('Outputs/H_vec_nrounds.csv', delimiter=",")
QAEP_scalability = np.genfromtxt('Outputs/H_QAEP_nrounds.csv', delimiter=",")
GEAT_scalability = np.genfromtxt('Outputs/H_GEAT_nrounds.csv', delimiter=",")

Hmin_scalability_var = np.genfromtxt('Outputs/Hmin_vec_nrounds_var.csv', delimiter=",")
H_scalability_var = np.genfromtxt('Outputs/H_vec_nrounds_var.csv', delimiter=",")
QAEP_scalability_var = np.genfromtxt('Outputs/H_QAEP_nrounds_var.csv', delimiter=",")
GEAT_scalability_var = np.genfromtxt('Outputs/H_GEAT_nrounds_var.csv', delimiter=",")

# Slices of entropies
H_slice_vs_alpha = np.genfromtxt('Outputs/H_slice_vs_alpha.csv', delimiter=",")
H_slice_vs_beta = np.genfromtxt('Outputs/H_slice_vs_beta.csv', delimiter=",")

# Read file into an array
H_test = np.genfromtxt('Outputs/H_array_cmap.csv', delimiter=",")
Hmin_test = np.genfromtxt('Outputs/Hmin_array_cmap.csv', delimiter=",")

H_array = np.genfromtxt('Outputs/H_array_100.csv', delimiter=",")
Hmin_array = np.genfromtxt('Outputs/Hmin_array_100.csv', delimiter=",")

H_array_is = np.genfromtxt('Outputs/H_array_100_is.csv', delimiter=",")
Hmin_array_is = np.genfromtxt('Outputs/Hmin_array_100_is.csv', delimiter=",")

# For Bootstrap error estimation
Hmin_datapoint_vtest = np.genfromtxt('Outputs/Hmin_datapoint_vec.csv', delimiter=",")
H_datapoint_vtest = np.genfromtxt('Outputs/H_datapoint_vec.csv', delimiter=",")
AEP_datapoint_vtest = np.genfromtxt('Outputs/AEP_datapoint_vec.csv', delimiter=",")
EAT_datapoint_vtest = np.genfromtxt('Outputs/EAT_datapoint_vec.csv', delimiter=",")

Hmin_datapoint_ns1e4 = np.genfromtxt('Outputs/Hmin_datapoint_vec_ns1e4.csv', delimiter=",")
H_datapoint_ns1e4 = np.genfromtxt('Outputs/H_datapoint_vec_ns1e4.csv', delimiter=",")
AEP_datapoint_ns1e4 = np.genfromtxt('Outputs/AEP_datapoint_vec_ns1e4.csv', delimiter=",")
EAT_datapoint_ns1e4 = np.genfromtxt('Outputs/EAT_datapoint_vec_ns1e4.csv', delimiter=",")

Hmin_datapoint_ns1e5 = np.genfromtxt('Outputs/Hmin_datapoint_vec_ns1e5.csv', delimiter=",")
H_datapoint_ns1e5 = np.genfromtxt('Outputs/H_datapoint_vec_ns1e5.csv', delimiter=",")
AEP_datapoint_ns1e5 = np.genfromtxt('Outputs/AEP_datapoint_vec_ns1e5.csv', delimiter=",")
EAT_datapoint_ns1e5 = np.genfromtxt('Outputs/EAT_datapoint_vec_ns1e5.csv', delimiter=",")

Hmin_datapoint_ns1e6 = np.genfromtxt('Outputs/Hmin_datapoint_vec_ns1e6.csv', delimiter=",")
H_datapoint_ns1e6 = np.genfromtxt('Outputs/H_datapoint_vec_ns1e6.csv', delimiter=",")
AEP_datapoint_ns1e6 = np.genfromtxt('Outputs/AEP_datapoint_vec_ns1e6.csv', delimiter=",")
EAT_datapoint_ns1e6 = np.genfromtxt('Outputs/EAT_datapoint_vec_ns1e6.csv', delimiter=",")

Hmin_datapoint_ns1e7 = np.genfromtxt('Outputs/Hmin_datapoint_vec_ns1e7.csv', delimiter=",")
H_datapoint_ns1e7 = np.genfromtxt('Outputs/H_datapoint_vec_ns1e7.csv', delimiter=",")
AEP_datapoint_ns1e7 = np.genfromtxt('Outputs/AEP_datapoint_vec_ns1e7.csv', delimiter=",")
EAT_datapoint_ns1e7 = np.genfromtxt('Outputs/EAT_datapoint_vec_ns1e7.csv', delimiter=",")


# Read file into an array
#Hmin_vec_a = np.genfromtxt('Outputs/Hmin_vec_a.csv', delimiter=",")
#H_vec_a = np.genfromtxt('Outputs/H_vec_a.csv', delimiter=",")
#H_QAEP_a = np.genfromtxt('Outputs/H_QAEP_a.csv', delimiter=",")
#H_GEAT_a = np.genfromtxt('Outputs/H_GEAT_a.csv', delimiter=",")

#Hmin_vec_b = np.genfromtxt('Outputs/Hmin_vec_b.csv', delimiter=",")
#H_vec_b = np.genfromtxt('Outputs/H_vec_b.csv', delimiter=",")
#H_QAEP_b = np.genfromtxt('Outputs/H_QAEP_b.csv', delimiter=",")
#H_GEAT_b = np.genfromtxt('Outputs/H_GEAT_b.csv', delimiter=",")