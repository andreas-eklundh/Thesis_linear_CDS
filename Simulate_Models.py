from Models.BaselineCIR_alternative.CIR_Multifactor import CIRIntensity as CIR
from Models.BaselineCIR_alternative.Gamma_solver import DeterministicGamma as Gamma_class 


from Models.LHCModels.LHC_single import LHC_single as LHC
from Models.LHCModels.LHC_single import get_CDS_Model, rebuild_lhc_struct, cds_value

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm 

### Simulate 1 LHC dataset with specific parameter Choises.
lhc = LHC(0.025,0.4,0.25)
Y_dim,m = 1,2
# Here, parameters are set already
rng = np.random.default_rng(1000)
X0 = 0.5
chi0 = np.array([1] + [X0]*m)
lhc.initialise_LHC(Y_dim,m,X0=X0,rng=rng)
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
T,M = 1, 100
# Use same seed to reproduce same randomness.
mat_grid = np.array([1,3,5,7,10]) # Typical maturity grid
# mat_grid = np.array([5]) 

n_mat = mat_grid.shape[0]
T_path, chi_Q,chi_P = lhc.simul_latent_states(chi0=chi0,T=T,M=M,n_mat=n_mat,seed=200)


# Holld maturity to be 5
t_mat_grid = np.ascontiguousarray(mat_grid[:, None] + T_path[None, :])   # shape (len(T_M_grid), len(t_obs))

# Doung this actual payment dates induces weird jumps. May be more correct though.
# mat_actual = np.array([[0.2137 + i, 0.4658 + i,0.7178 + i,0.9671+i ] 
#                         for i in range(0,int(np.max(t_mat_grid)+1))]).flatten()
# # Ensure mat_actual is sorted
# mat_actual_sorted = np.sort(mat_actual)
# # For each element in t_mat_grid, find the smallest mat_actual that is >= element
# t_mat_grid = np.array([mat_actual_sorted[np.searchsorted(mat_actual_sorted, val, side='left')] 
#                                 for val in t_mat_grid.flatten()]).reshape(t_mat_grid.shape)

t0 = T_path
# mat_actual = np.array([[0.2137 + i, 0.4658 + i,0.7178 + i,0.9671+i ] 
#                         for i in range(0,int(np.max(t0)+1))]).flatten()
# # Ensure mat_actual is sorted
# mat_actual_sorted = np.sort(mat_actual)
# # For each element in t_mat_grid, find the smallest mat_actual that is >= element
# t0 = np.array([mat_actual_sorted[np.searchsorted(mat_actual_sorted, val, side='left')] 
#                                 for val in t0.flatten()]).reshape(t0.shape)
kappa, theta, gamma1 = lhc.kappa,lhc.theta,lhc.gamma1[0]
r = lhc.r
Y_dim = lhc.Y_dim
delta = lhc.delta
tenor = lhc.tenor 
lhc_numba = rebuild_lhc_struct(kappa, theta, gamma1, r, Y_dim, delta, tenor)


# Draw noise vector:
save_path = "./Exploratory/"   # <--- change to your path
color_cycle = plt.cm.tab10.colors  

R = norm.rvs(size = (t_mat_grid.shape[0]*t_mat_grid.shape[1]),scale = lhc.sigma_err).reshape(t_mat_grid.shape) # simulate at beginning - faster!
CDS_simul = get_CDS_Model(T_path, t0, t_mat_grid, chi_Q.T, lhc_numba)  # + R
# Test of value:
print(cds_value(lhc_numba, t0[0], t0[0], t_mat_grid[:,0],CDS_simul[:,0]) @ chi_Q[0,:])

print(np.min(CDS_simul),np.max(CDS_simul))

# Reestimation of the Process. Try Kalman filter on the recreated CDS spreads.
# Kalman automatically initiates at random. 
lhc_kalman_params,  Xn,Zn, Pn= lhc.run_n_kalmans(T_path, t_mat_grid, CDS_simul.T,base_seed=2000,n_restarts=5)
# Recalculate CDS spreads, if value approach:
CDS_kalman = Zn # Latent states directly from filter.
# Try and calculate anyway. Provide X and Y.
#CDS_kalman = lhc.CDS_model(T_path, t_mat_grid, CDS_simul.T,X=Xn[:,1:].T, Y=Xn[:,0])



# ## Option pricing Kalman.   
# t_start = 1
# T_option = t_start + 5
# strike_spreads = np.array([50,100,150]) / 10000
# n_poly = np.array([1,5,30])
# ### Pricing in standard lhc. Model alredy defines 
# price_strikes_lhcK = np.zeros(strike_spreads.shape[0])
# chi0 = Xn[-1,:]
# # Set sigma to be 0.3. Only one thats gonna be used. 
# # Should be sparked up in case of price for higher strikes. 
# for idx,k in enumerate(strike_spreads):
#     price_strikes_lhcK[idx] = lhc.get_cdso_pric_MC(t=0,t0=t_start,t_M=T_option,
#                              strike=k, chi0=chi0, N=500,M=1000)


lhc_params = lhc.optimal_parameter_set(t_obs=T_path,T_M_grid=t_mat_grid,CDS_obs=CDS_simul.T,base_seed=1000,n_restarts=1)
# utilizing the previous vals
X, Y, Z = lhc.get_states(T_path, t_mat_grid, CDS_simul.T)
# This way to get CDS spreads is consistent with the article -> do that here. 
CDS_model = lhc.CDS_model(T_path, t_mat_grid, CDS_simul.T)

np.set_printoptions(precision=4, suppress=True)  # fewer decimals, no scientific notation
print(f'Optimal parameters, Kalman: {lhc_kalman_params}')
print(f'Optimal Parameters Filipovic {lhc_params}')
print(f'Actual Parameters: {params_actual}')


# # Test plotting with Kalman estimates included
save_path = "./Exploratory/"   # <--- change to your path
color_cycle = plt.cm.tab10.colors  

# --- Latent states plots ---
n_states = chi_Q.shape[1]  # total number of latent states

for i in range(0,n_states):
    fig, ax = plt.subplots(figsize=(10,4))
    
    # Custom name for the first latent state
    state_name = "Y / Survival Process" if i == 0 else f"X{i}"
    
    # Plot simulated Q vs P - only X
    ax.plot(T_path, chi_Q[:, i], "--", alpha=0.8, label=f"{state_name}, Q (sim)", color="blue")
    ax.plot(T_path, chi_P[:, i], "-", alpha=0.8, label=f"{state_name}, P (sim)", color="red")
    
    # Plot Kalman estimate
    ax.plot(T_path, Xn[:, i], "-", alpha=0.9, label=f"{state_name} (Kalman)", color="green")
    # Optional: Filipovic solution if available
    if i == 0:
        ax.plot(T_path, Y, "-", alpha=0.9, label=f"{state_name} (Filipovic)", color="yellow")
    else:
        ax.plot(T_path, X[i-1, :], "-", alpha=0.9, label=f"{state_name} (Filipovic)", color="yellow")
    
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
    ax.plot(T_path, CDS_kalman[:, i], "--", alpha=0.9, label=f"CDS Kalman, T_mat={mat_grid[i]}", color='green')
    ax.plot(T_path, CDS_model[:, i], "--", alpha=0.9, label=f"CDS Filipovic, T_mat={mat_grid[i]}", color='yellow')

ax.set_xlabel("Time (years)")
ax.set_ylabel("CDS Spreads")
ax.set_title("CDS Spreads Comparison")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(save_path, "SimulLHC_CDS_Spreads.png"), dpi=150)
plt.close(fig)


# ### Option pricing. 
# ### Pricing in standard lhc. Model alredy defines 
# price_strikes_lhc = np.zeros(strike_spreads.shape[0])
# chi0 = np.append([Y[-1]],X[:,-1])
# # Set sigma to be 0.3. Only one thats gonna be used. 
# # Should be sparked up in case of price for higher strikes. 
# P_params = np.array([1]*lhc.m + [1]*lhc.m +  [0.03]*lhc.m + [7e-5]  )
# for idx,k in enumerate(strike_spreads):
#     price_strikes_lhc[idx] = lhc.get_cdso_pric_MC(t=0,t0=t_start,t_M=T_option,
#                              strike=k, chi0=chi0, N=500,M=1000,P_params=P_params)


# print(f'CDSO prices, LHC-Kalman: {price_strikes_lhcK*10000} Bps')
# print(f'CDSO prices, LHC: {price_strikes_lhc*10000} Bps')





#### CIR SIMULATION. ######
# These parameters satisfy the necessary.
X_dim=2
cir = CIR(0.0252, 0.4, 0.25,X_dim)
cir.set_params(params=None, seed=100)
print(cir.kappa,cir.theta,cir.sigma,cir.kappa_p,cir.theta_p,cir.sigma_err)
params_cir = np.concatenate([cir.kappa,cir.theta,cir.sigma,cir.kappa_p,cir.theta_p,cir.sigma_err])
# Simulate. We are using an Euler discretization. 
T,M = 1, 500
# Simulate: 
np.random.seed(11)
# Set initial lambda to the one we would get in a LHC model.
lambda0 = 0.04 #  gamma1* X0 / 1 
# This is under Q 
T_return,lambda_mil_Q = cir.simulate_intensity(lambda0=lambda0,T=T,M=M,scheme="Milstein",measure='Q')
T_return,lambda_mil_P = cir.simulate_intensity(lambda0=lambda0,T=T,M=M,scheme="Milstein", measure='P')

t_mat_grid = np.ascontiguousarray(mat_grid[:, None] + T_return[None, :])   # shape (len(T_M_grid), len(t_obs))
# mat_actual = np.array([[0.2137 + i, 0.4658 + i,0.7178 + i,0.9671+i ] 
#                         for i in range(0,int(np.max(t_mat_grid)+1))]).flatten()
# # Ensure mat_actual is sorted
# mat_actual_sorted = np.sort(mat_actual)
# # For each element in t_mat_grid, find the smallest mat_actual that is >= element
# t_mat_grid = np.array([mat_actual_sorted[np.searchsorted(mat_actual_sorted, val, side='left')] 
#                                 for val in t_mat_grid.flatten()]).reshape(t_mat_grid.shape)

CDS_cir = np.ones((t_mat_grid.T.shape))
for i in range(t_mat_grid.shape[1]):
    lambda_curr  = np.array([lambda_mil_Q[i]])
    mat_curr = t_mat_grid[:,i]
    CDS_cir[i,:] = cir.cds_spread(lambda_curr,params,T_return[i],mat_curr)
    print(f'Done with {(i+1)/t_mat_grid.shape[1]} %')

### See if possible to estimate the parameters. For that, need to turn into integrated lambda.
# Need to generate the deterministic lambdas.
# We dont need to callibrate deterministic lambda. We can simply take log of the solution to Ricatti eqs.
model = Gamma_class(r, delta, tenor)
t_mats = np.concatenate(([0],mat_grid))
Gamma = np.zeros((CDS_cir.shape[0], T_return.shape[0]))
cali_params = np.zeros((CDS_cir.shape[0], mat_grid.shape[0]))
for t_idx in range(CDS_cir.shape[0]):
    # Calibrate back hazard rates
    t_grid_payments = np.array([tenor*i for i in range(int(np.max(mat_grid)/tenor)+1)])   
    
    cali_params[t_idx, : ] = model.calibrate_deterministic(CDS_cir[t_idx,:] , mat_grid, 0.0, t_grid_payments)
    # Generate the survial probabilities/survival process
    for i in range(T_return.shape[0]):
        Gamma[t_idx,i] = model.Gamma_fun(cali_params[t_idx, : ] ,T_return[i],t_mats)
        
Gamma_kalman = Gamma[:,np.isin(T_return, mat_grid).flatten()]

# Select only params at maturity. 
param_cir_est, Xn_cir,Zn_cir, Pn_cir = cir.run_n_kalman(T_return,t_mat_grid,Gamma_kalman,n_restarts=5)

# # Then get CDS spread
CDS_cir_model = np.ones((t_mat_grid.T.shape))
for i in range(t_mat_grid.shape[1]):
    lambda_curr  = np.array([Xn_cir[i,:]])
    mat_curr = t_mat_grid[:,i]
    CDS_cir_model[i,:] = cir.cds_spread(lambda_curr,params,T_return[i],mat_curr)
    print(f'Done with {(i+1)/t_mat_grid.shape[1]} %')


# print(f'Optimal Parameters CIR {param_cir_est}')
# print(f'Actual Parameters: {params_cir}')

# --- Latent state (simulated vs Kalman estimate) ---
save_path = "./Exploratory/"   # <--- change to your path
color_cycle = plt.cm.tab10.colors  

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6), sharey=False)

ax1.plot(T_return, lambda_mil_Q, "-", alpha=0.7, label="Simulated X1 (CIR), Q", color="blue")
ax1.plot(T_return, lambda_mil_P, "-", alpha=0.7, label="Simulated X1 (CIR), P", color="blue")

ax1.plot(T_return, Xn_cir[:,0], "--", alpha=0.9, label="Kalman Estimated X1", color="red")

ax1.set_xlabel("Time (years)")
ax1.set_ylabel("Latent State")
ax1.legend()
ax1.set_title("Latent State")

# --- CDS spreads (simulated vs Kalman estimates) ---
for i in range(CDS_cir.shape[1]):
    color = color_cycle[i % len(color_cycle)]
    ax2.plot(T_return, CDS_cir[:,i], "-", alpha=0.7,
             label=f"CDS Sim, T_mat={mat_grid[i]}", color=color)
    ax2.plot(T_return, CDS_cir_model[:,i], "--", alpha=0.9,
             label=f"CDS Kalman, T_mat={mat_grid[i]}", color='blue')

ax2.set_xlabel("Time (years)")
ax2.set_ylabel("CDS Spreads")
ax2.legend()
ax2.set_title("Spreads")

fig.tight_layout()
fig.savefig(os.path.join(save_path, "SimulCIR_states_vs_Kalman.png"), dpi=150)
plt.close(fig)


#### CIR option pricing.
price_strikes_CIR = np.zeros(strike_spreads.shape[0])
chi0 = Xn_cir[-1,:]
# Set sigma to be 0.3. Only one thats gonna be used. 
# Should be sparked up in case of price for higher strikes. 
lambda_process = np.sum(Xn_cir, axis=1)

for idx,k in enumerate(strike_spreads):
    price_strikes_CIR[idx] = cir.get_cdso_pric_MC(param_cir_est, t=0,t0=t_start,t_M=T_option,
                             strike=k, lambda0=lambda_process[-1], N=500,M=1000)

print(f'CDSO prices Kalman: {price_strikes_CIR*10000} Bps')




cir.mc

test = 1