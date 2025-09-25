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


# mat_grid = np.array([1,2,3,4,5,7,10,15,20,30])
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
# CDS_obs = np.array(test_df[['1Y','2Y','3Y','4Y','5Y','7Y','10Y','15Y',
#                    '20Y','30Y']])
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
optim_params,  Xn,Zn, Pn= lhc.run_n_kalmans(t, t_mat_grid, CDS_obs,n_restarts=1)
kappa, theta, gamma1 = optim_params[:lhc.m],optim_params[lhc.m:2*lhc.m], optim_params[2*lhc.m]

default_intensity = lhc.default_intensity(Xn[:,1:].T,Xn[:,0])
mpr,girsanov = lhc.get_MPR(optim_params,Xn[:,0],Xn[:,1:].T,CDS_obs)

np.savez("C:/Users/andre/OneDrive/KU, MAT-OEK/Kandidat/Thesis/Thesis_linear_CDS/Results/Kalman_resultsLHC.npz",
         final_param=optim_params,
         Xn=Xn[:,1:],
         Yn=Xn[:,0],
         Default_intensity = default_intensity,
         CDS_model = Zn,
         MPR = mpr)


import matplotlib.pyplot as plt
import os
### Test: Compare To CDS. 

save_path = f"./Exploratory/"   # <--- change to your path
os.makedirs(save_path, exist_ok=True)

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(t, Xn[:,0], "-", alpha=0.7, label=f"Survival Process")

ax.set_xlabel("Time (years)")
ax.set_ylabel("Survival process")
ax.set_title("Kalman CDS States")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(save_path, "SurvivalKalman_LHC.png"), dpi=150)
plt.close(fig)


# MPR

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(t, mpr, "-", alpha=0.7, label=f"MPR")

ax.set_xlabel("Time (years)")
ax.set_ylabel("Market Price of Risk")
ax.set_title("Market price of risk")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(save_path, "MPR_LHC.png"), dpi=150)
plt.close(fig)


# Default intensity: 
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(t, default_intensity , "-", alpha=0.7, label=f"Default Process")

ax.set_xlabel("Time (years)")
ax.set_ylabel("Default Intensity")
ax.set_title("Kalman CDS States")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(save_path, "DefaultKalman_LHC.png"), dpi=150)
plt.close(fig)


color_cycle = plt.cm.tab10.colors  # or use ax._get_lines.prop_cycler

fig, ax = plt.subplots(figsize=(10,6))
for i in range(1,Xn.shape[1]):
    color = color_cycle[i % len(color_cycle)] 
    ax.plot(t, Xn[:,i], "-", alpha=0.7, label=f"X{i}",color=color)

ax.set_xlabel("Time (years)")
ax.set_ylabel("States")
ax.set_title("Kalman CDS States")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(save_path, "StatesKalman_LHC.png"), dpi=150)
plt.close(fig)


### Test: Compare To CDS.   
color_cycle = plt.cm.tab10.colors  # or use ax._get_lines.prop_cycler

fig, ax = plt.subplots(figsize=(10,6))
for i in range(Zn.shape[1]):
    color = color_cycle[i % len(color_cycle)] 

    #ax.plot(t, pred_Zn[i,:].flatten(), label=f"Predicted CDS Spread, Maturity {mat_grid[i]}")
    ax.plot(t, CDS_obs[:, i], "o", alpha=0.7,color=color, label=f"Observed {mat_grid[i]}Y CDS")
    ax.plot(t, Zn[:,i].flatten(), "-", linewidth=2,color=color, label=f"Predicted Kalman, {mat_grid[i]}Y CDS")
ax.set_xlabel("Time (years)")
ax.set_ylabel("CDS Spread")
ax.set_title("Kalman CDS Spreads")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(save_path, "CDSKalman_LHC.png"), dpi=150)
plt.close(fig)



### States plotting under p and Q
color_cycle = plt.cm.tab10.colors  # or use ax._get_lines.prop_cycler

fig, ax = plt.subplots(figsize=(10,6))
for i in range(1,Xn.shape[1]):
    color = color_cycle[i % len(color_cycle)] 
    ax.plot(t, Xn[:,i], "-", alpha=0.7, label=f"X{i}, P",color=color)
    # Currently specified under P. Need to add to  
    ax.plot(t, (Xn[:,i] - girsanov.T).flatten(), "--", alpha=0.7, label=f"X{i}, Q",color=color)

ax.set_xlabel("Time (years)")
ax.set_ylabel("States")
ax.set_title("Kalman CDS States")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(save_path, "States_QP_Kalman_LHC.png"), dpi=150)
plt.close(fig)


test = 1


