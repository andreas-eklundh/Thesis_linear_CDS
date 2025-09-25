import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
t_grid = [0 + 0.25* i for i in range(0, int(10 / 0.25)+1)]


#### Preliminary investigation. 
test_df = pd.read_excel("./Data/test_data.xlsx")

# Pivot
test_df = test_df.pivot(index = ['Date','Ticker'],
                        columns='Tenor',values = 'Par Spread').reset_index()
# Test on subset data ownly to get very few obs. One large spread increase to test.
#test_df = test_df[(test_df['Date']<'2021-01-01') & (test_df['Date']>='2019-06-01')]
test_df = test_df[5::5]

# Function to convert tenors to months to same metric (so
test_df['Years']= ((test_df['Date'] - test_df['Date'].min()).dt.total_seconds() / (365.25 * 24 * 3600)).drop_duplicates()
t = np.array(test_df['Years'])
# These are all available time points. 


# mat_grid = np.array([1,2,3,4,5,7,10])
# mat_grid = np.array([2,3,4,5,7,10])
mat_grid = np.array([5]) # Matures in 5 years, but specific dates.
t_mat_grid = np.ascontiguousarray(mat_grid[:, None] + t[None, :])   # shape (len(T_M_grid), len(t_obs))

# For a quarterly CDS, following effective payment dates.¨ (do actual calcs at some point)
# So actual termination dates.
# Needs to be modified when generalized.
mat_actual = np.array([[0.2137 + i, 0.4658 + i,0.7178 + i,0.9671+i ] 
                        for i in range(0,int(np.max(t_mat_grid)+1))]).flatten()
# Ensure mat_actual is sorted
mat_actual_sorted = np.sort(mat_actual)

# For each element in t_mat_grid, find the smallest mat_actual that is >= element
t_mat_grid = np.array([mat_actual_sorted[np.searchsorted(mat_actual_sorted, val, side='left')] 
                                for val in t_mat_grid.flatten()]).reshape(t_mat_grid.shape)


# So effective date at first possible date above 
# This is the typical grid
# CDS_obs = np.array(test_df[['1Y','2Y','3Y','4Y','5Y','7Y','10Y']])
# CDS_obs = np.array(test_df[['2Y','3Y','4Y','5Y','7Y','10Y']])
CDS_obs = np.array(test_df[['5Y']]) 

from Models.LHCModels.LHC_single import LHC_single
#from Models.Extended.LHC_Temp import LHC_single


# Defaults? How are thesy done?

## Some initial guesses. 
# LHC(2) model - gamma dictates. Y 1-dim, X 2-dim. 
# Write function to do initial guess. Need dim of y and X
lhc = LHC_single( r=0.0252,delta=0.4,cds_tenor= 0.25 )
# initialise guesses for params. 
# set Y_dim=1, X_dim=1 to test remaining logic, X_dim>1 for general purposes.
# Why? X_dim=1 easy to solve problem if using only one spread. 
lhc.initialise_LHC(Y_dim=1,X_dim=1,rng=None)

### TODO: Try to implement a totally basic exapmple CF 40.
# lhc.optimize_params(t_obs=t,T_M_grid=t_mat_grid,CDS_obs=CDS_obs)


# Test several random points. 
out_params= lhc.optimal_parameter_set(t_obs=t,T_M_grid=t_mat_grid,CDS_obs=CDS_obs,n_restarts=1)

## Should include a function for optimizing base on say 5 differetn random points or more.
# And then take the best.
# This is likely what they do in paper.

#temp_param = lhc.flatten_params()

#lhc.unflatten_params(np.array([0.5881308 , 0.89771373, 0.00151587, 0.28225612, 0.5871308 ]))


import os

# After optimize, define lhc model for inputting.
# ---- Get states ----
X, Y, Z = lhc.get_states(t, t_mat_grid, CDS_obs)
S = Y  # rename Y to S

default_intensity = lhc.default_intensity(X,Y)

# ---- Model CDS ----
CDS_model = lhc.CDS_model(t, t_mat_grid, CDS_obs)
np.savez("C:/Users/andre/OneDrive/KU, MAT-OEK/Kandidat/Thesis/Thesis_linear_CDS/Results/LHC_results.npz",
         final_param=out_params,
         Xn=X,
         Yn=Y,
         Default_intensity = default_intensity,
         CDS_model = CDS_model)



# ---- Save folder ----
save_path = f"./Exploratory/"   # <--- change to your path
os.makedirs(save_path, exist_ok=True)
# Reuse the same figure object to overwrite plots

# ---- Plot states ----
# Only states X2,X3,... as X1 is proportional to default intensity in this model specification.
fig, ax = plt.subplots(figsize=(10,6))
for i in range(1,X.shape[0]):
    ax.plot(t, X[i,:], label=f"X{i+1}")
ax.set_xlabel("Time (years)")
ax.set_ylabel("State values")
ax.set_title("Latent States (X)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(save_path, "states_test.png"), dpi=150)
plt.close(fig)

# ---- Plot survival ----
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(t, S, label="S (Y)")
ax.set_xlabel("Time (years)")
ax.set_ylabel("State values")
ax.set_title("Latent States (Y) i.e. Survival Process")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(save_path, "survival_test.png"), dpi=150)
plt.close(fig)

# ---- Plot CDS ----

fig, ax = plt.subplots(figsize=(10,6))
tenors = mat_grid

color_cycle = plt.cm.tab10.colors  # or use ax._get_lines.prop_cycler

for i, tenor in enumerate(tenors):
    color = color_cycle[i % len(color_cycle)]  # pick one consistent color
    ax.plot(t, CDS_obs[:, i], "o", alpha=0.7, color=color, label=f"Observed {tenor}Y CDS")
    ax.plot(t, CDS_model[:, i], "-", linewidth=2, color=color, label=f"Model {tenor}Y CDS")

ax.set_xlabel("Time (years)")
ax.set_ylabel("CDS Spread")
ax.set_title("Observed vs Model CDS for Multiple Tenors")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(save_path, "cds_multi_tenors.png"), dpi=150)
plt.close(fig)

# ---- Plot default intensity ----
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(t, default_intensity, label="Default intensity")
ax.set_xlabel("Time (years)")
ax.set_ylabel("Default Intensity")
ax.set_title("Default intensity model")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(save_path, "default_intensity_test.png"), dpi=150)
plt.close(fig)


# Optimization succeeded, params [0.5881308  0.00152638 0.5871308 ], seed 20. 
# Optimization succeeded, params [0.18940583 0.00140457 0.18840583], seed 1950

# Payoff tests. 

t_start = 1
T_option = t_start + 5
strike_spreads = np.array([250,300,350]) / 10000
n_poly = np.array([1,5,30])

# Get time zero stats. 
M = 1000 # (how continous we make the plot)
for k in strike_spreads:
    fig, ax = plt.subplots(figsize=(10,6))
    # Get lower bounds on z.
    b_min,b_max = lhc.get_bBounds(t_start,T_option,k)
    # Create some grid fr plotting,
    plot_grid = np.array([b_min + i*(b_max-b_min)/M for i in range(0,M+1)])
    for n in n_poly:
        Y_t = Y[0] # Should maybe even use y after for forecasts?
        price = np.array([lhc.PriceCDS(z,n,t=0,t0=t_start,t_M=T_option,k=k,Y=Y_t) for z in plot_grid])
        ax.plot(plot_grid,price, label=f"Price CDSO, n={n}")
        print(f"Done with n={n},k={k}")
    ax.set_xlabel("z")
    ax.set_ylabel("CDSO price")
    ax.set_title(f"Estimated CDSO price, k={k}")
    ax.legend()
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f"CDSO_price_k_{k}.png"), dpi=150)
    plt.close(fig)

test = 1