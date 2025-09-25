import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Get parameters and corresponding states. 
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

CDS_obs = np.array(test_df[['5Y']]) 


# CIR Baseline.
data = np.load("./Results/kalman_resultsCIR.npz")
final_paramCIR = data["final_param"]
XnCIR = data["Xn"]
ZnCIR = data["Zn"]
PnCIR = data["Pn"]

# LHC Baseline
data = np.load("./Results/LHC_results.npz")
final_paramLHC = data["final_param"]
XnLHC = data["Xn"]
YnLHC=data["Yn"]
ZnLHC = data["CDS_model"]
Default_intensityLHC = data["Default_intensity"]


data = np.load("./Results/kalman_resultsLHC.npz")
final_paramLHCK = data["final_param"]
XnLHCK = data["Xn"]
YnLHCK=data["Yn"]
ZnLHCK = data["CDS_model"]
Default_intensityLHCK = data["Default_intensity"]
mprLHCK = data["MPR"]



# Do the plotting. 
save_path = f"./Results/DANBNK/"   # <--- change to your path

# Default intensities
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(t, Default_intensityLHCK , "-", alpha=0.7, color='green', label=f"LHC Kalman")
ax.plot(t, Default_intensityLHC , "-", alpha=0.7, color='blue', label=f"LHC")
ax.plot(t, XnCIR , "-", alpha=0.7, color='red', label=f"CIR Kalman")


ax.set_xlabel("Time (years)")
ax.set_ylabel("Default Intensity")
ax.set_title("Default intensities in different models, Danske Bank")
ax.legend()
ax.grid()
fig.tight_layout()
fig.savefig(os.path.join(save_path, "DefaultIntensities.png"), dpi=150)
plt.close(fig)



# Recreated Spreads
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(t, ZnLHCK , "-", alpha=0.7, color='green', label=f"LHC Kalman")
ax.plot(t, ZnLHC , "-", alpha=0.7, color='blue', label=f"LHC")
ax.plot(t, ZnCIR , "-", alpha=0.7, color='red', label=f"CIR Kalman")
ax.plot(t, CDS_obs , "o", alpha=0.7, color='black', label=f"Observations")

ax.grid()
ax.set_xlabel("Time (years)")
ax.set_ylabel("Model Spreads Intensity")
ax.set_title("Model Spreads in different models, Danske Bank")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(save_path, "Spreads.png"), dpi=150)
plt.close(fig)


# Survival process.

YnCIR = np.exp(-np.cumsum(XnCIR*(t[1]-t[0]))) # only approximates
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(t, YnLHCK , "-", alpha=0.7, color='green', label=f"LHC Kalman")
ax.plot(t, YnLHC , "-", alpha=0.7, color='blue', label=f"LHC")
ax.plot(t, YnCIR , "-", alpha=0.7, color='red', label=f"CIR Kalman")

ax.grid()
ax.set_xlabel("Time (years)")
ax.set_ylabel("Survival prob Intensity")
ax.set_title("Survival probabilities in different models, Danske Bank")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(save_path, "SurvivalProb.png"), dpi=150)
plt.close(fig)
