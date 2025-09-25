from Models.BaselineCIR.CIRIntensity import CIRIntensity as CIR
from Models.BaselineCIR.CIRIntensity import cds_spread 
from Models.BaselineCIR.CIRIntensity import CIR_Theta as CIR_Theta

from Models.LHCModels.LHC_single import LHC_single as LHC
from Models.LHCModels.LHC_single import get_CDS_Model, rebuild_lhc_struct 

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm 

### Simulate 1 LHC dataset with specific parameter Choises.
lhc = LHC(0.025,0.4,0.25)
Y_dim,m = 1,2
# Here, parameters are set already
rng = np.random.default_rng(1000)
lhc.initialise_LHC(Y_dim,m,rng=rng)
lhc.flatten_params()
params = lhc.flatten_params()
lhc.unflatten_params(params[:2*m+1])
# Set also P params. 
lhc_P = lhc.build_P_params(rng=rng)
print(f"constraints Q: {lhc.gamma1 - lhc.kappa + lhc.kappa*lhc.theta}")
print(f"constraints P: {lhc.gamma1 - lhc.kappa_p + lhc.kappa_p*lhc.theta_p}")
params_actual = [lhc.kappa, lhc.theta,lhc.gamma1,lhc.kappa_p,lhc.theta_p,lhc.sigma, lhc.sigma_err]
print(lhc.kappa, lhc.theta,lhc.gamma1,lhc.kappa_p,lhc.theta_p,lhc.sigma, lhc.sigma_err)

# Simulate. We are using an Euler discretization. 
# Start at 0.5 also for aloowing for more jump op and down. Again, likly too large initial cov
Chi0 = np.array([1]+[0.5]*m)
T,M = 1, 500
# Use same seed to reproduce same randomness. Two different BMs, tied by MPR now. Also simulate without problem
T_path, chi_P = lhc.simul_latent_states(Chi0,T=T,M=M, measure='P',seed=1000)
T_path, chi_Q = lhc.simul_latent_states(Chi0,T=T,M=M, measure='Q',seed=1001)



# Holld maturity to be 5
mat_grid = np.array([5]) 
t_mat_grid = np.ascontiguousarray(mat_grid[:, None] + T_path[None, :])   # shape (len(T_M_grid), len(t_obs))
mat_actual = np.array([[0.2137 + i, 0.4658 + i,0.7178 + i,0.9671+i ] 
                        for i in range(0,int(np.max(t_mat_grid)+1))]).flatten()
# Ensure mat_actual is sorted
mat_actual_sorted = np.sort(mat_actual)
# For each element in t_mat_grid, find the smallest mat_actual that is >= element
t_mat_grid = np.array([mat_actual_sorted[np.searchsorted(mat_actual_sorted, val, side='left')] 
                                for val in t_mat_grid.flatten()]).reshape(t_mat_grid.shape)

t0 = T_path
mat_actual = np.array([[0.2137 + i, 0.4658 + i,0.7178 + i,0.9671+i ] 
                        for i in range(0,int(np.max(t0)+1))]).flatten()
# Ensure mat_actual is sorted
mat_actual_sorted = np.sort(mat_actual)
# For each element in t_mat_grid, find the smallest mat_actual that is >= element
t0 = np.array([mat_actual_sorted[np.searchsorted(mat_actual_sorted, val, side='left')] 
                                for val in t0.flatten()]).reshape(t0.shape)
kappa, theta, gamma1 = lhc.kappa,lhc.theta,lhc.gamma1[0]
r = lhc.r
Y_dim = lhc.Y_dim
delta = lhc.delta
tenor = lhc.tenor 
lhc_numba = rebuild_lhc_struct(kappa, theta, gamma1, r, Y_dim, delta, tenor)


# Draw noise vector:
R = norm.rvs(size = (t_mat_grid.shape[0]*t_mat_grid.shape[1]),scale = lhc.sigma_err).reshape(t_mat_grid.shape) # simulate at beginning - faster!
CDS_simul = get_CDS_Model(T_path, t0, t_mat_grid, chi_Q.T, lhc_numba) + R

print(np.min(CDS_simul),np.max(CDS_simul))

# Reestimation of the Process. Try Kalman filter on the recreated CDS spreads.
# Kalman automatically initiates at random. 
optim_params,  Xn,Zn, Pn= lhc.run_n_kalmans(T_path, t_mat_grid, CDS_simul.T,n_restarts=1)
out_params= lhc.optimal_parameter_set(t_obs=T_path,T_M_grid=t_mat_grid,CDS_obs=CDS_simul.T,n_restarts=1)
X, Y, Z = lhc.get_states(T_path, t_mat_grid, CDS_simul.T)
CDS_model = lhc.CDS_model(T_path, t_mat_grid, CDS_simul.T)


print(f'Optimal parameters, Kalman: {optim_params}')
print(f'Optimal Parameters Filipovic {out_params}')
print(f'Actual Parameters: {params_actual}')


# Test plotting with Kalman estimates included
save_path = "./Exploratory/"   # <--- change to your path
color_cycle = plt.cm.tab10.colors  

# --- Latent states plots ---
n_states = chi_Q.shape[1]  # total number of latent states

for i in range(n_states):
    fig, ax = plt.subplots(figsize=(10,4))
    
    # Custom name for the first latent state
    state_name = "Y / Survival Process" if i == 0 else f"X{i}"
    
    # Plot simulated Q vs P
    ax.plot(T_path, chi_Q[:, i], "--", alpha=0.8, label=f"{state_name}, Q (sim)", color="blue")
    ax.plot(T_path, chi_P[:, i], "-", alpha=0.8, label=f"{state_name}, P (sim)", color="red")
    
    # Plot Kalman estimate
    ax.plot(T_path, Xn[:, i], "-", alpha=0.9, label=f"{state_name} (Kalman)", color="green")
    
    # Optional: Filipovic solution if available
    if i < X.shape[0]:
        ax.plot(T_path, X[i, :], "-", alpha=0.9, label=f"{state_name} (Filipovic)", color="yellow")
    
    ax.set_xlabel("Time (years)")
    ax.set_ylabel(state_name)
    ax.set_title(f"Latent State: {state_name}")
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f"SimulLHC_State_{i}.png"), dpi=150)
    plt.close(fig)


# --- CDS spreads plot ---
fig, ax = plt.subplots(figsize=(10,5))

for i in range(CDS_simul.shape[0]):
    ax.plot(T_path, CDS_simul[i,:], "-", alpha=0.7, label=f"CDS Sim, T_mat={mat_grid[i]}", color='black')
    ax.plot(T_path, Zn[:, i], "--", alpha=0.9, label=f"CDS Kalman, T_mat={mat_grid[i]}", color='green')
    ax.plot(T_path, CDS_model[:, i], "--", alpha=0.9, label=f"CDS Filipovic, T_mat={mat_grid[i]}", color='yellow')

ax.set_xlabel("Time (years)")
ax.set_ylabel("CDS Spreads")
ax.set_title("CDS Spreads Comparison")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(save_path, "SimulLHC_CDS_Spreads.png"), dpi=150)
plt.close(fig)




#### CIR SIMULATION. ######

kappa_cir,theta_cir,sigma_cir = 1.65e-01, 1.10e-01, 0.08
kappa_p_cir,theta_p_cir, sigma_err_cir = 2.64e-01, 3.7e-01, 8.21e-05 # sigma error needs to be very small.
params_cir = np.array([kappa_cir,theta_cir,sigma_cir,kappa_p_cir,theta_p_cir,sigma_err_cir])
# These parameters satisfy the necessary.
cir = CIR(0.0252, 0.4, 0.25)
cir.set_params(params_cir)

# Simulate. We are using an Euler discretization. 
# T,M = 10, 500
# Simulate: 
np.random.seed(11)
# Set initial lambda to the one we would get in a LHC model.
lambda0 = gamma1* Chi0[1] / Chi0[1]
# This is under Q 
T_return,lambda_mil = cir.simulate_intensity(lambda0=lambda0,T=T,M=M,scheme="Milstein")
# Estimate. 
cir_n = CIR_Theta(cir.kappa,cir.theta,cir.sigma,
                        cir.kappa_p,cir.theta_p,
                        cir.delta,cir.tenor,cir.r)

CDS_cir = np.ones((t_mat_grid.T.shape))
for i in range(t_mat_grid.shape[1]):
    lambda_curr  = np.array([lambda_mil[i]])
    mat_curr = t_mat_grid[:,i]
    CDS_cir[i,:] = cds_spread(cir_n,lambda_curr,T_return[i],mat_curr)


param_cir, Xn,Zn, Pn = cir.run_n_Kalman(T_return,t_mat_grid,CDS_cir,n_optim=5)


print(f'Optimal Parameters CIR {param_cir}')
print(f'Actual Parameters: {params_cir}')

# --- Latent state (simulated vs Kalman estimate) ---
save_path = "./Exploratory/"   # <--- change to your path
color_cycle = plt.cm.tab10.colors  

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6), sharey=False)

ax1.plot(T_return, lambda_mil, "-", alpha=0.7, label="Simulated X1 (CIR)", color="blue")
ax1.plot(T_return, Xn[:,0], "--", alpha=0.9, label="Kalman Estimated X1", color="red")

ax1.set_xlabel("Time (years)")
ax1.set_ylabel("Latent State")
ax1.legend()
ax1.set_title("Latent State")

# --- CDS spreads (simulated vs Kalman estimates) ---
for i in range(CDS_cir.shape[1]):
    # color = color_cycle[i % len(color_cycle)]
    ax2.plot(T_return, CDS_cir[:,i], "-", alpha=0.7,
             label=f"CDS Sim, T_mat={mat_grid[i]}", color='black')
    ax2.plot(T_return, Zn[:,i], "--", alpha=0.9,
             label=f"CDS Kalman, T_mat={mat_grid[i]}", color='blue')

ax2.set_xlabel("Time (years)")
ax2.set_ylabel("CDS Spreads")
ax2.legend()
ax2.set_title("Spreads")

fig.tight_layout()
fig.savefig(os.path.join(save_path, "SimulCIR_states_vs_Kalman.png"), dpi=150)
plt.close(fig)


test = 1