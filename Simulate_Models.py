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
T,M = 1, 50
# Use same seed to reproduce same randomness.
# mat_grid = np.array([1,3,5,7,10]) # Typical maturity grid
mat_grid = np.array([3,5,10]) 

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
lhc_kalman_params,  Xn,Zn, Pn= lhc.run_n_kalmans(T_path, t_mat_grid, CDS_simul.T,base_seed=3000,n_restarts=1)
Xn_kalman,Yn_kalman = lhc.kalman_X_Y(T_path,Xn)
# Recalculate CDS spreads, if value approach:
# Get Kalman States


CDS_kalman = Zn # Latent states directly from filter.
# Try and calculate anyway. Provide X and Y.
# CDS_kalman = lhc.CDS_model(T_path, t_mat_grid, CDS_simul.T,X=Xn[:,1:].T, Y=Xn[:,0])



## Option pricing Kalman.   
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


# lhc_params = lhc.optimal_parameter_set(t_obs=T_path,T_M_grid=t_mat_grid,CDS_obs=CDS_simul.T,base_seed=3000,n_restarts=1)
# # utilizing the previous vals
# X, Y, Z = lhc.get_states(T_path, t_mat_grid, CDS_simul.T)
# # This way to get CDS spreads is consistent with the article -> do that here. 
# CDS_model = lhc.CDS_model(T_path, t_mat_grid, CDS_simul.T)

np.set_printoptions(precision=4, suppress=True)  # fewer decimals, no scientific notation
print(f'Optimal parameters, Kalman: {lhc_kalman_params}')
# print(f'Optimal Parameters Filipovic {lhc_params}')
print(f'Actual Parameters: {params_actual}')


# # # Test plotting with Kalman estimates included
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
    
    # Optional: Filipovic solution if available
    if i == 0:
        ax.plot(T_path, Yn_kalman, "-", alpha=0.9, label=f"{state_name} (Kalman)", color="green")
        # ax.plot(T_path, Y, "-", alpha=0.9, label=f"{state_name} (Filipovic)", color="yellow")
    else:
        # ax.plot(T_path, X[i-1, :], "-", alpha=0.9, label=f"{state_name} (Filipovic)", color="yellow")
        # Plot Kalman estimate
        ax.plot(T_path, Xn_kalman[:, i-1], "-", alpha=0.9, label=f"{state_name} (Kalman)", color="green")

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
    # ax.plot(T_path, CDS_model[:, i], "--", alpha=0.9, label=f"CDS Filipovic, T_mat={mat_grid[i]}", color='yellow')

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
# params_actual =np.array([-0.1115+0.2247,0.2247*0.0611/(-0.1115+0.2247),0.0702,0.2247,0.0611,0.003]) # Use params from a article
cir.set_params(params=None, seed=100)
print(cir.kappa,cir.theta,cir.sigma,cir.kappa_p,cir.theta_p,cir.sigma_err)
params_cir = np.concatenate([cir.kappa,cir.theta,cir.sigma,cir.kappa_p,cir.theta_p,cir.sigma_err])
# Simulate. We are using an Euler discretization. 
T,M = 1, 50
# Simulate: 
np.random.seed(11)
# Set initial lambda to the one we would get in a LHC model.
lambda0 = np.ones(X_dim)*0.01 #  gamma1* X0 / 1 
# This is under Q 
T_return,lambda_mil_Q = cir.simulate_intensity(lambda0=lambda0,T=T,M=M,scheme="Euler",measure='Q')
T_return,lambda_mil_P = cir.simulate_intensity(lambda0=lambda0,T=T,M=M,scheme="Euler", measure='P')

mat_grid = np.array([1,3,5,7,10])
# mat_grid = np.array([5])
t_mat_grid = np.ascontiguousarray(mat_grid[:, None] + T_return[None, :])   # shape (len(T_M_grid), len(t_obs))

# Get true CDS spreads. 
CDS_cir = np.ones((t_mat_grid.T.shape))
for i in range(t_mat_grid.shape[1]):
    lambda_curr  = np.array([lambda_mil_Q[i]])
    mat_curr = t_mat_grid[:,i]
    CDS_cir[i,:] = cir.cds_spread(lambda_curr,params_cir,T_return[i],mat_curr)

### See if possible to estimate the parameters. For that, need to turn into integrated lambda.
# Need to generate the deterministic lambdas.
# We dont need to callibrate deterministic lambda. We can simply take log of the solution to Ricatti eqs.
model = Gamma_class(r, delta, tenor)
t_mats = np.concatenate(([0],mat_grid))
extrapolate_grid = np.array([i  for i in range(int(np.max(mat_grid))+ 1)])    

Gamma = np.zeros((CDS_cir.shape[0], extrapolate_grid.shape[0]))
cali_params = np.zeros((CDS_cir.shape[0], mat_grid.shape[0]))

for t_idx in range(CDS_cir.shape[0]):
    # Calibrate back hazard rates
    t_grid_payments = np.array([tenor*i for i in range(int(np.max(mat_grid)/tenor)+1)])   
    cali_params[t_idx, : ] = model.calibrate_deterministic(CDS_cir[t_idx,:] , mat_grid, 0.0, t_grid_payments)
    # Generate the survial probabilities/survival process
    for i in range(extrapolate_grid.shape[0]):
        Gamma[t_idx,i] = model.Gamma_fun(cali_params[t_idx, : ],extrapolate_grid[i],t_mats)

# Survival CIR Kalman model:
survival = np.zeros((CDS_cir.shape[0], extrapolate_grid.shape[0]))
for t_idx in range(CDS_cir.shape[0]):    
    for i in range(extrapolate_grid.shape[0]):
        survival[t_idx,i] = np.exp(-Gamma[t_idx,i] )


Gamma_kalman = Gamma[:,np.isin(extrapolate_grid, mat_grid).flatten()]


def compute_gamma_matrix(T_return, lambda_mil_Q, tenors, extrapolate_last=True):
    T_return = np.asarray(T_return)
    lambda_mil_Q = np.asarray(lambda_mil_Q)
    tenors = np.asarray(tenors)

    # dt array (works for nonuniform grids)
    dt = np.diff(T_return, prepend=T_return[0]).reshape((lambda_mil_Q.shape[0],1))   # dt[0] will be 0

    # cumulative integral up to each grid point (integral from T_return[0] to T_return[i])
    Gamma_cum = np.cumsum(lambda_mil_Q * dt)

    # Build all target absolute times: T_return[:,None] + tenors[None,:]
    n = T_return.size
    m = tenors.size
    targets = (T_return[:, None] + tenors[None, :]).ravel()  # shape (n*m,)

    # Interpolate cumulative integral at each target time:
    # np.interp returns Gamma_cum value at target by linear interpolation on xp=T_return.
    # For targets beyond T_return[-1], decide how to handle: default right=Gamma_cum[-1]
    Gamma_at_targets = np.interp(targets, T_return, Gamma_cum, left=Gamma_cum[0], right=Gamma_cum[-1])
    Gamma_at_targets = Gamma_at_targets.reshape(n, m)

    # If you want linear extrapolation beyond last grid using last lambda value:
    if extrapolate_last:
        # find targets beyond last time
        last_t = T_return[-1]
        last_idx = (targets > last_t).reshape(n, m)
        if np.any(last_idx):
            # incremental contribution beyond last grid:
            extra = (targets.reshape(n, m) - last_t) * lambda_mil_Q[-1]
            Gamma_at_targets[last_idx] = Gamma_cum[-1] + extra[last_idx]

    # Gamma from t_i to t_i + tenor_j is Gamma_at_targets - Gamma_cum[i]
    Gamma_stoch = Gamma_at_targets - Gamma_cum[:, None]

    # enforce non-negativity due to numerical noise:
    Gamma_stoch = np.maximum(Gamma_stoch, 0.0)

    return Gamma_stoch

# Gamma_stoch = compute_gamma_matrix(T_return, lambda_mil_Q, tenors=mat_grid)

# Select only params at maturity. 
params_cir_est, Xn_cir,Zn_cir, Pn_cir = cir.run_n_kalman(T_return,t_mat_grid,Gamma_kalman,n_restarts=10)

# # Then get CDS spread
CDS_cir_model = np.ones((t_mat_grid.T.shape))
for i in range(t_mat_grid.shape[1]):
    lambda_curr  = np.array([Xn_cir[i,:]])
    mat_curr = t_mat_grid[:,i]
    CDS_cir_model[i,:] = cir.cds_spread(lambda_curr,params_cir_est,T_return[i],mat_curr)
    print(f'Done with {(i+1)/t_mat_grid.shape[1]} %')

np.set_printoptions(precision=4, suppress=True)
print(f'Optimal Parameters CIR {params_cir_est}')
print(f'Actual Parameters: {params_cir}')

# --- Latent state (simulated vs Kalman estimate) ---

save_path = "./Exploratory/"   # <--- change to your path
color_cycle = plt.cm.tab10.colors  

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6), sharey=False)

# === LATENT STATES (all X[:,i]) ===
n_states = Xn_cir.shape[1]  # number of latent factors

for i in range(n_states):
    color = color_cycle[i % len(color_cycle)]
    ax1.plot(T_return, Xn_cir[:, i], "--", alpha=0.9, color=color, label=f"Kalman X{i+1}")

# Optionally add simulated versions if available (like lambda_mil_Q / lambda_mil_P)
if 'lambda_mil_Q' in locals():
    ax1.plot(T_return, lambda_mil_Q, "-", alpha=0.7, label="Simulated X1 (CIR, Q)", color="black")
if 'lambda_mil_P' in locals():
    ax1.plot(T_return, lambda_mil_P, "-", alpha=0.7, label="Simulated X1 (CIR, P)", color="gray")

ax1.set_xlabel("Time (years)")
ax1.set_ylabel("Latent State")
ax1.legend()
ax1.set_title("Latent States")

# === CDS spreads (simulated vs Kalman estimates) ===
for i in range(CDS_cir.shape[1]):
    color = color_cycle[i % len(color_cycle)]
    ax2.plot(T_return, CDS_cir[:, i], "-", alpha=0.7,
             label=f"CDS Sim, T_mat={mat_grid[i]}", color=color)
    ax2.plot(T_return, CDS_cir_model[:, i], "--", alpha=0.9,
             label=f"CDS Kalman, T_mat={mat_grid[i]}", color=color)

ax2.set_xlabel("Time (years)")
ax2.set_ylabel("CDS Spreads")
ax2.legend()
ax2.set_title("Spreads")

fig.tight_layout()

# Save figure
os.makedirs(save_path, exist_ok=True)
fig.savefig(os.path.join(save_path, "SimulCIR_states_vs_Kalman.png"), dpi=150)
plt.close(fig)
#### CIR option pricing.
# price_strikes_CIR = np.zeros(strike_spreads.shape[0])
# chi0 = Xn_cir[-1,:]
# # Set sigma to be 0.3. Only one thats gonna be used. 
# # Should be sparked up in case of price for higher strikes. 
# lambda_process = np.sum(Xn_cir, axis=1)

# for idx,k in enumerate(strike_spreads):
#     price_strikes_CIR[idx] = cir.get_cdso_pric_MC(param_cir_est, t=0,t0=t_start,t_M=T_option,
#                              strike=k, lambda0=lambda_process[-1], N=500,M=1000)

# print(f'CDSO prices Kalman: {price_strikes_CIR*10000} Bps')





test = 1