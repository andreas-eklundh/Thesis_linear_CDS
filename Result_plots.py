import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from Models.Utils import global_fit_measures

# Get parameters and corresponding states. 
test_df = pd.read_excel("./Data/test_data.xlsx")

# Pivot
test_df = test_df.pivot(index = ['Date','Ticker'],
                        columns='Tenor',values = 'Par Spread').reset_index()
# Test on subset data ownly to get very few obs. One large spread increase to test.
#test_df = test_df[(test_df['Date']<'2021-01-01') & (test_df['Date']>='2019-06-01')]
# test_df = test_df[5::5]

# Function to convert tenors to months to same metric (so
test_df['Years']= ((test_df['Date'] - test_df['Date'].min()).dt.total_seconds() / (365.25 * 24 * 3600)).drop_duplicates()
t = np.array(test_df['Years'])

CDS_obs = np.array(test_df[['1Y','3Y','5Y','7Y','10Y']])


# CIR Baseline.
data = np.load("./Results/Kalman_resultsCIR.npz")
final_paramCIR = data["final_param"]
XnCIR = data["Xn"]
ZnCIR = data["Zn"]
PnCIR = data["Pn"]
YnCIR = data["Yn"]
Default_intensityCIR = data["default_intensity"]
CDS_CIR = data["CDS_cir"]



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
ax.plot(t, Default_intensityCIR , "-", alpha=0.7, color='red', label=f"CIR Kalman")


ax.set_xlabel("Time (years)")
ax.set_ylabel("Default Intensity")
ax.set_title("Default intensities in different models, Danske Bank")
ax.legend()
ax.grid()
fig.tight_layout()
fig.savefig(os.path.join(save_path, "DefaultIntensities.png"), dpi=150)
plt.close(fig)



# Recreated Spreads
import matplotlib.pyplot as plt
import os

# Example list of maturities (adjust names if your columns differ)
maturities = [1, 3, 5, 7, 10]  # or ['1Y','3Y','5Y','7Y','10Y']

# Loop through maturities and make a separate plot for each
for m,mat in enumerate(maturities):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(t, ZnLHCK[:,m], "-", alpha=0.7, color='green', label="LHC Kalman")
    ax.plot(t, ZnLHC[:,m], "-", alpha=0.7, color='blue', label="LHC")
    ax.plot(t, CDS_CIR[:,m], "-", alpha=0.7, color='red', label="CIR Kalman")
    ax.plot(t, CDS_obs[:,m], "o", alpha=0.5, color='black', label="Observations")

    ax.grid(True)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Model Spreads / Intensity")
    ax.set_title(f"Model Spreads at {m}-year maturity (Danske Bank)")
    ax.legend()

    fig.tight_layout()
    save_file = os.path.join(save_path, f"Spreads_{m}Y.png")
    fig.savefig(save_file, dpi=150)
    plt.close(fig)


# Survival process.
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




#### Compute Global measures of fit. 
# Stack together CDS frames

models = [ZnLHC,ZnLHCK,CDS_CIR] # stacked fitted CDS spreads. 

gfm = global_fit_measures(CDS_obs, models)

rmse_series, rmse = gfm.rmse() 
ape_series, ape = gfm.ape() 
aae_series, aae = gfm.aae() 
arpe_series, arpe = gfm.arpe() 

# Example structure:
cols_names = ["LHC", "LHC Kalman", "CIR"]

# Your NumPy arrays: (n_obs, n_models)
# e.g. rmse_series.shape == (T, 3)
# and global scalars/vectors: rmse, ape, aae, arpe each (n_models,)

# --- Wrap arrays into DataFrames ---
rmse_series = pd.DataFrame(rmse_series, columns=cols_names)
ape_series  = pd.DataFrame(ape_series,  columns=cols_names)
aae_series  = pd.DataFrame(aae_series,  columns=cols_names)
arpe_series = pd.DataFrame(arpe_series, columns=cols_names)

metrics = [
    ("RMSE", rmse_series),
    ("APE", ape_series),
    ("AAE", aae_series),
    ("ARPE", arpe_series)
]

# === 1) 4-panel figure with all models ===
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes = axes.flatten()

for ax, (label, df) in zip(axes, metrics):
    for col in df.columns:
        ax.plot(test_df['Date'], df[col], label=col, lw=1.5)
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_xlabel("Observation")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)

# Add legend (outside for clarity)
axes[-1].legend(title="Models", loc="upper right", bbox_to_anchor=(1.25, 1.0))

fig.suptitle("Risk Measure Time Series by Model", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(save_path, "Global_fit_errors.png"), dpi=150)
plt.close(fig)

# === 2) Global summary table ===
# If rmse, ape, aae, arpe are 1D arrays/lists
global_summary = pd.DataFrame({
    "RMSE": np.ravel(rmse),
    "APE": np.ravel(ape),
    "AAE": np.ravel(aae),
    "ARPE": np.ravel(arpe)
}, index=cols_names)

print("\nGlobal Risk Measure Summary:\n")
print(global_summary.round(6))

stpper = 1