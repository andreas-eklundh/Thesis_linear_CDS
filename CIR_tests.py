import numpy as np 
from scipy.optimize import minimize, NonlinearConstraint, Bounds 

from numba import njit, float64, int64 
from Models.ATSMGeneral.ATSM import ATSM
from scipy.stats import norm, ncx2, gamma, expon
from scipy.integrate import quad

from numba.experimental import jitclass 
from Models.BaselineCIR_alternative.CIR_Multifactor import CIRIntensity

r = 0.0252
delta = 0.4
tenor = 0.25
X_dim = 2

cir = CIRIntensity(r,delta,tenor,X_dim)
x0 = np.array([0])
import pandas as pd
t_grid = [0 + 0.25* i for i in range(0, int(10 / 0.25)+1)]
# Test of grid

# _get_default_grid(0.5,t_grid)    

#### Preliminary investigation. 
test_df = pd.read_excel("./Data/test_data.xlsx")

# Pivot
test_df = test_df.pivot(index = ['Date','Ticker'],
                        columns='Tenor',values = 'Par Spread').reset_index()
# Test on subset data ownly to get very few obs. One large spread increase to test.
test_df = test_df[::5]
test_df['Years']= ((test_df['Date'] - test_df['Date'].min()).dt.total_seconds() / (365.25 * 24 * 3600)).drop_duplicates()
t = np.array(test_df['Years']) # t of the CDS.
CDS_obs = np.array(test_df[['1Y','3Y','5Y','7Y','10Y']])
# Read in inferred survival probs.
data = np.load("C:/Users/andre/OneDrive/KU, MAT-OEK/Kandidat/Thesis/Thesis_linear_CDS/Gamma_Calibration/CDS_TS_plot.npz")
t_mats_plots = data['t_mats_plots']
survival=data['survival']
Gamma = data['Gamma']
default_prob = data['default_prob']

# Find corresponding entries in t_mats_plots.
# Note, we need to retrieve the cols/rows corresponding to the dates for each day. 
# mat_grid = np.array([1,3,5,7,10])
mat_grid = np.array([1,5,10] ) #,5,7,10])

# The running working maturities
t_mat_grid = np.ascontiguousarray(mat_grid[:, None] + t[None, :])   # shape (len(T_M_grid), len(t_obs))

t_mats_plots_kalman = t_mats_plots[np.isin(t_mats_plots,mat_grid).flatten()]

survival_kalman = survival[::5,np.isin(t_mats_plots, mat_grid).flatten()]
Gamma_kalman = Gamma[::5,np.isin(t_mats_plots, mat_grid).flatten()]


# Negative process. Multiply by -1 everywhere. Let A and a get these too. 
# params, Xn,Zn,Pn = cir.run_kalman_filter(t,t_mat_grid,Y=Gamma_kalman ,seed=1000)

# Ttry with several restarts. 
params, Xn,Zn,Pn = cir.run_n_kalman(t,t_mat_grid,Y=Gamma_kalman,base_seed=1000,n_restarts=3)

# Set new optimal parameters too.
cir.set_params(params)
# default_prob_model= np.exp(Zn)

# Save values:

# With Params in place, we can utilize CIR class to do pricing, simulations etc. 
CDS_cir = np.zeros(survival_kalman.shape)
for n in range(survival_kalman[:,0].shape[0]):    
    CDS_cir[n,:] = cir.cds_spread(Xn[n,:],params,t[n],t_mat_grid[:,n])
    print(f'Done with {(n+1)/survival_kalman[:,0].shape[0]} %')

# Compare.
import os
import matplotlib.pyplot as plt
save_path = "./Exploratory/"   # <--- change to your path
os.makedirs(save_path, exist_ok=True)
plt.figure(figsize=(12, 6))

for j, T in enumerate(mat_grid):
    # Observed line
    plt.plot(t, CDS_obs[:, j], 
                label=f"Obs {T}Y", marker='o', alpha=0.5)
    # Implied line
    plt.plot(t, CDS_cir[:, j], 
                label=f"CIR {T}Y", linestyle='--',)

plt.xlabel("Time")
plt.ylabel("CDS Spread (bps)")
plt.title("Observed vs Model (CIR) CDS Spreads by Maturity")
plt.legend(ncol=2)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(save_path, f"CIRSpreads_impliedGamma.png"), dpi=150)
plt.close()

## Latent states: 
for i in range(0,X_dim):
    fig, ax = plt.subplots(figsize=(10,4))
    
    # Custom name for the first latent state
    state_name = f"X{i}"
    
    ax.plot(t,Xn[:, i], "--", alpha=0.8, label=f"{state_name}", color="blue")
    
    ax.set_xlabel("Time (years)")
    ax.set_ylabel(state_name)
    ax.set_title(f"Latent State: {state_name}")
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f"CIR_State_{i}.png"), dpi=150)
    plt.close(fig)


# Default intensity probability since inception (identical to state if dim=1)
default_intensity = np.sum(Xn,axis=1)
fig, ax = plt.subplots(figsize=(10,4))

# Custom name for the first latent state
state_name = f"lambda(t)"

ax.plot(t,default_intensity, "--", alpha=0.8, label=f"{state_name}", color="blue")

ax.set_xlabel("Time (years)")
ax.set_ylabel(state_name)
ax.set_title(f"Latent State: {state_name}")
ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(save_path, f"CIR_intensity_{i}.png"), dpi=150)
plt.close(fig)

# Get induced default probability: 
Yn = np.exp(-np.cumsum(default_intensity*(t[1]-t[0]))) # only approximates

state_name = f"S(t)"
fig, ax = plt.subplots(figsize=(10,4))

ax.plot(t,Yn, "--", alpha=0.8, label=f"{state_name}", color="blue")

ax.set_xlabel("Time (years)")
ax.set_ylabel(state_name)
ax.set_title(f"Survival process")
ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(save_path, f"CIR_survival_{i}.png"), dpi=150)
plt.close(fig)


np.savez("C:/Users/andre/OneDrive/KU, MAT-OEK/Kandidat/Thesis/Thesis_linear_CDS/Results/Kalman_resultsCIR.npz",
        final_param=params,
        Xn=Xn,
        Zn=Zn,
        Pn = Pn,
        Yn = Yn,
        default_intensity = default_intensity,
        CDS_cir = CDS_cir)


test = 1