from Models.BaselineCIR_alternative.CIR_Multifactor import CIRIntensity as CIR
from Models.BaselineCIR_alternative.Gamma_solver import DeterministicGamma as Gamma_class 
from Models.LHCModels.LHC_single import LHC_single as LHC
from Models.LHCModels.LHC_single import get_CDS_Model, rebuild_lhc_struct, cds_value, solve_mu1, compute_stationary
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm 

## Some global parameters (simulate forward in time, grid fineness.)
import numpy as np
import matplotlib.pyplot as plt
from Models.BaselineCIR_alternative.CIR_Multifactor import CIRIntensity as CIR
from Models.BaselineCIR_alternative.Gamma_solver import DeterministicGamma as Gamma_class 
from Models.LHCModels.LHC_single import LHC_single as LHC
from Models.LHCModels.LHC_single import get_CDS_Model, rebuild_lhc_struct, cds_value, solve_mu1, compute_stationary
import os
from scipy.stats import norm

# === Simulation setup ===
import numpy as np
import matplotlib.pyplot as plt
import os

from Models.LHCModels.LHC_single import LHC_single as LHC

# === Global setup ===
T = 5
M_grid = [100, 1000, int(1e4),int(1e5),int(1e6)]
schemes = ['Euler', 'Milstein']   # row 0 = Euler, row 1 = Milstein
vars_map = {'X': 1, 'Y': 0, 'Z': 'Z'}  # mapping for convenience

# === Initialize LHC once (parameters fixed) ===
lhc = LHC(0.025, 0.4, 0.25)
Y_dim, m = 1, 2
rng = np.random.default_rng(500)
X0 = 0.3
chi0 = np.array([1] + [X0] * m)
lhc.initialise_LHC(Y_dim, m, X0=X0, rng=rng)
lhc.flatten_params()
params = lhc.flatten_params()
lhc.unflatten_params(params[:2 * m + 1])
_ = lhc.build_P_params(rng=rng)

print(f'Params {params}')


mat_grid = np.array([1, 3, 5, 7, 10])
n_mat = mat_grid.shape[0]

os.makedirs("./Simulation_studies", exist_ok=True)

# === Helper: run sims and return time + data for a given scheme and M ===
def simulate_for(scheme, M):
    # latent states (X_Y shape: time x (1 + m) or similar)
    T_path, X_Y = lhc.simul_latent_states(chi0=chi0, T=T, M=M, n_mat=n_mat, seed=40, scheme=scheme)
    # make sure shapes: X_Y[:,0] = Y, X_Y[:,1:] = X components
    Y_true = X_Y[:, 0]
    X_true = X_Y[:, 1:]        # shape (n_t, m)
    # transformed Z from X/Y
    # Avoid division by zero: small eps
    Z_trans = np.array([X_true[i, :] / (Y_true[i]) for i in range(Y_true.shape[0])])
    # simulate Z directly and transform back to X,Y via kalman_X_Y
    T_path, Z_true = lhc.simul_Z(chi0=chi0[1:], T=T, M=M, n_mat=n_mat, seed=40, scheme=scheme)
    X_trans, Y_trans = lhc.kalman_X_Y(T_path, Z_true)
    # X_trans shape: (n_t, m) ; Y_trans: (n_t,)
    return T_path, X_true, Y_true, Z_true, X_trans, Y_trans, Z_trans

# === For each variable (X, Y, Z) create its own 2x3 figure ===
for var in ['X', 'Y', 'Z']:
    fig, axes = plt.subplots(2, 5, figsize=(18, 8), sharex=True)
    fig.suptitle(f"Simulation discretization comparison — {var}", fontsize=16)

    for col_idx, M in enumerate(M_grid):
        for row_idx, scheme in enumerate(schemes):
            # simulate
            (T_path,
             X_true, Y_true, Z_true,
             X_trans, Y_trans, Z_trans) = simulate_for(scheme, M)

            ax = axes[row_idx, col_idx]
            ax.grid(True)

            # Plot depending on var
            if var == 'Y':
                ax.plot(T_path, Y_true, linestyle='-', marker='', label='Y True', color='red')
                ax.plot(T_path, Y_trans, linestyle='--', marker='', label='Y Transformed', color='blue')
                ax.set_ylabel('Y value')
            elif var == 'X':
                # if multiple X components, plot first component (or you can plot all)
                ax.plot(T_path, X_true, linestyle='-', marker='', label='X True', color='red')
                ax.plot(T_path, X_trans, linestyle='--', marker='', label='X Transformed', color='blue')
                ax.set_ylabel('X value')
            elif var == 'Z':
                ax.plot(T_path, Z_true, linestyle='-', marker='', label='Z True', color='red')
                ax.plot(T_path, Z_trans, linestyle='--', marker='', label='Z Transformed', color='blue')
                ax.set_ylabel('Z value (comp 0)')

            # Titles and labels
            if row_idx == 0:
                ax.set_title(f"M = {M}    (scheme = {scheme})")
            else:
                ax.set_title(f"(scheme = {scheme})")
            ax.set_xlabel('Time')

            # Legend only once per column to avoid clutter
            ax.legend(loc='best', fontsize='small')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = f"./Simulation_studies/Simulation_discretization_{var}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


test = 0