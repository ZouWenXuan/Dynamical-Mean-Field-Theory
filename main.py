# =============================================================================
# main test
# =============================================================================

from model.DMFT_numeric import DMFT_rate_model
from model.RNN import rate_model
import matplotlib.pyplot as plt
import numpy as np

#%% Define the parameters
# neural struture
N = 2000
g = 0.2
sigma = 0.0
eta = 0.5
nl = 'Tanh'

# simulation
T = 20
dt = 0.1
nIte_list = [30, 10, 20]
nTraj_list = [1000, 10000, 100000]  
damp = 0.7
threshold = 1e-10

# initialzed
dist = "uniform"
scale = 1


DMFT1 = DMFT_rate_model(g, eta, sigma, nl, T, dt, nTraj_list, nIte_list,\
                        damp, threshold, dist, scale)
RNN1 = rate_model(N, g, eta, sigma, nl, T, dt, dist, scale)   


#%% Comparison (example)
# Run integral and loop
mx1, mp1, Cx1, Cp1, Chix1, Chip1 = RNN1.compute_observables()   
mx2, mp2, Cx2, Cp2, Chix2, Chip2, Traj_dict = DMFT1.main_loop(record_traj=True) 

# Plot comparison
plt.figure(figsize=(5,4), dpi=200)
plt.plot(np.diag(Cx1), label='numerical', lw=4)
plt.plot(np.diag(Cx2), label='DMFT', ls='--', lw=4)
plt.legend()
plt.show()

plt.figure(figsize=(5,4), dpi=200)
plt.plot(mx1, label='numerical', lw=4)
plt.plot(mx2, label='DMFT', ls='--', lw=4)
plt.legend()
plt.show()

#%% Plot
# Plot Parameter
t_step = int(T/dt)
t = np.arange(0,T,0.1)

# load data for main figure
m1, m2 = mp1[0:t_step], mp2[0:t_step]
C1, C2 = Cp1[0:t_step, 0:t_step], Cp2[0:t_step, 0:t_step]

# Plot comparison
plt.figure(figsize=(6,4), dpi=500)
plt.plot(t,np.diag(C1), label=r'$C(t,t)$', lw=3.5, c='lightgreen')
plt.plot(t,np.diag(C2), ls='--', lw=2.5, c='k', alpha=0.7)
plt.plot(t,mx1, label=r'$m(t)$', lw=3.5, c='steelblue')
plt.plot(t,mx2, label='DMFT', ls='--', lw=2.5, c='k', alpha=0.7)
plt.legend(loc=1)
plt.xlabel(r'$t$ (ms)', size=15)
plt.ylabel('Observables', size=15)
plt.xticks(size=15)
plt.yticks(size=15)

