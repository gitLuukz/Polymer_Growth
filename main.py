"""
Main document of the project
"""


import numpy as np
import time
import plot_functions as pf
import functions as f

"""
Initialisation of the project
Generate parameters and initial positions, forces and velocity
"""
N = 250
epsilon = 0.25
sig = 0.8 
kb = 1 
T = 1 
d = 1 
Group_size = 1000


R_save = np.zeros((2,N,Group_size))
distance_save = np.zeros((1,N,Group_size))
w_save = np.zeros((1,N-2,Group_size))
W_save = np.zeros((1,N-2,Group_size))
P_save = np.zeros((N-2,Group_size))
Rpol_avg = np.zeros((N,1))

#%% Rosenbluth
"""
Simulating the polymer with the Rosenbluth algorithm
"""
start_time = time.clock()
R_save, distance_save,w_save,W_save = f.Iterative_simulation( Group_size, N, epsilon, sig, R_save, distance_save, w_save,W_save)
print('rosenbluth')
print(time.clock()-start_time)
P_save = np.cumprod(W_save[0,:,:],0)

Rpol_avg = f.modify(P_save, Rpol_avg, N, Group_size, distance_save)
sigma = f.sigma(N,P_save,Group_size,distance_save**2)

a = f.fit(N,Rpol_avg[:,0]**2)
fit = (a[0]*(np.linspace(1,N,N)-1)**0.75)**2

pf.chain(R_save[:,:,2], 5)
fignr = 1
pf.length(Rpol_avg, N, sigma, fit, fignr, "Rosenbluth")

#%% Perm
"""
Simulating the polymer with the PERM algorithm 
"""

# R_perm = np.zeros((2,N,1))
# W_perm = np.zeros((N-2,1))
# P_perm = np.zeros((N-2,1))

# iteration_perm = 1000
# start_time = time.clock()
# R_perm, W_perm, distance_PERM = f.run_PERM(R_perm,W_perm, iteration_perm, epsilon, sig, N)
# print('PERM')
# print(time.clock()-start_time)
# P_perm = np.cumprod(W_perm[:,:],0)
# Rpol_avg_p = np.zeros((N,1))
# Rpol_avg_p = f.modify_p(P_perm,Rpol_avg_p,N, R_perm.shape[2], distance_PERM)
# sigma_p = f.sigma_p(N,P_perm,R_perm.shape[2],distance_PERM**2)

# a_p = f.fit(N,Rpol_avg_p[:,0]**2)
# fit_p = (a_p[0]*(np.linspace(1,N,N)-1)**0.75)**2

# fignr = 2
# pf.length(Rpol_avg_p, N, sigma_p, fit_p, fignr, "PERM")