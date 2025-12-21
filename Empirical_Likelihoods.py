############## INVESTIGATION OF LIKELIHOODS ##############
#### Purpose: investigate how the likelihood region looks as a function of the parameters.
import numpy as np
from Models.LHCModels.LHC_single import LHC_single, kalman_wrapper,build_P_params,rebuild_lhc_struct
import pandas as pd
import os
###### Create grids.
from Models.LHCModels.LHC_single import rebuild_lhc_struct, get_CDS_Model, nonlinear_constraints
from Models.BaselineCIR_alternative.CIR_Multifactor import CIRIntensity as CIR
from Models.BaselineCIR_alternative.Gamma_solver import DeterministicGamma as Gamma_class
from scipy.stats import norm
import copy
import math
import seaborn as sns
import matplotlib.pyplot as plt

# ============= INITIALIZE CIR ============= #
# ### Simulate 1 LHC dataset with specific parameter Choises.
if __name__ == '__main__':
    sub_df = pd.read_excel("./Data/subset_data.xlsx")
    for firm in ['DANBNK','MONTE']:

        test_df = sub_df[(sub_df['Ticker']==firm)]
        test_df = test_df.pivot(index = ['Date','Ticker'],
                                columns='Tenor',values = 'Par Spread').reset_index()
        # Test on subset data ownly to get very few obs. One large spread increase to test.
        test_df['Years']= ((test_df['Date'] - test_df['Date'].min()).dt.total_seconds() / (365.25 * 24 * 3600)).drop_duplicates()

        t = np.array(test_df['Years'])

        mat_grid = np.array([1,2,3,4,5,7,10])
        t_mat_grid = np.ascontiguousarray(mat_grid[:, None] + t[None, :])   # shape (len(T_M_grid), len(t_obs))

        # Forwrard fill again. Back fill in case any initial missing
        CDS_obs = np.array(test_df[['1Y','2Y','3Y','4Y','5Y','7Y','10Y']].ffill().bfill())


        X_dim =1
        T,M = 10, 1000

        # ll,params_cir_kalman , Xn,Zn,Pn,se = cir.run_kalman_filter(T_return, t_mat_grid, Gamma_kalman_noise)
        # Read in the fitted kalman filter instead
        directory = f"./Results/{firm}"

        filepath = os.path.join(directory, f"Kalman_resultsCIR_Xdim{X_dim}.npz")
        data = np.load(filepath)
        final_param = data["final_param"]
        cir = CIR(0.00248,0.4,0.25,X_dim,cascading=True)
        # Here, parameters are set already
        seed = 4000
        r = cir.r
        delta = cir.delta
        tenor = cir.tenor
        cir.set_params(params=final_param)


        data = np.load(f"./Gamma_Calibration/{firm}/Data_{firm}.npz")
        t_mats_plots = data['t_mats_plots']
        survival=data['survival']
        Gamma = data['Gamma']
        default_prob = data['default_prob']
        gamma_hist = data['gamma_hist']
        
        t_mats_plots_kalman = t_mats_plots[np.isin(t_mats_plots,mat_grid).flatten()]

        survival_kalman = survival[:,np.isin(t_mats_plots, mat_grid).flatten()]
        Gamma_kalman = Gamma[:,np.isin(t_mats_plots, mat_grid).flatten()]
        Gamma_kalman_scale =Gamma_kalman #/ mat_grid[None, :]

        # assuming lhc.flatten_params() returns [kappa, theta, gamma1, sigma, lambda_i, sigma_err]
        # true_params = {
        #     "kappa": params_cir[0],
        #     "theta": params_cir[1],
        #     "sigma": params_cir[2],
        #     "lambda1": params_cir[3],
        #     "sigma_err": params_cir[4]
        # }

        kalman_params = {
            "kappa": final_param[0],
            "theta": final_param[1],
            "sigma": final_param[2],
            "lambda1": final_param[3],
            "sigma_err": final_param[4]
        }


        # Define relative/absolute spreads for scanning
        grid_spread = {
            "kappa": 0.05,
            "theta": 0.05,
            "sigma": 0.05,
            "lambda1": 0.1,     # absolute
            "sigma_err": 0.05
        }

        n_points = 20
        grids = {}
        for key, val in kalman_params.items():
            spread = grid_spread[key]
            val = float(val)  # ensure scalar
            if key == "lambda1":
                # min_bound = - cir.theta*cir.kappa
                max_bound = cir.kappa
                max_bound = np.minimum(max_bound-0.001,val + spread)
                grids[key] = np.linspace(val - spread,max_bound, n_points)
            else:
                grids[key] = np.linspace(max(1e-6, val * (1 - spread)), val * (1 + spread), n_points)

        for key in grids:
            if (key != "lambda1") :
                grids[key] = np.clip(grids[key], 1e-6, None)

        # --- Kalman evaluation setup ---

        results = []

        # --- Main scan loop ---
        for param_name, grid_values in grids.items():
            print(f"--- Investigating parameter: {param_name} ---")

            for val in grid_values:
                pvals = kalman_params.copy()
                pvals[param_name] = val

                # Build param vector: [kappa, theta, gamma1, lambda_i, sigma, sigma_err]
                x0_Q = np.concatenate([np.atleast_1d(pvals["kappa"]),
                                    np.atleast_1d(pvals["theta"])])
                x0_P = np.concatenate([np.atleast_1d(pvals["sigma"]),
                                    np.atleast_1d(pvals["lambda1"]),
                                    np.atleast_1d(pvals["sigma_err"])])
                params = np.concatenate([x0_Q, x0_P])

                # Evaluate likelihood
                # Set new params:
                cir.set_params(params)
                # Check if feller satisfied. if not flag and color differently.
                if np.any(cir.feller_constraint(params) <0):
                    print('Feller Condition Failed')
                    neg_log_like = np.nan

                else:
                    neg_log_like = cir.Kalman(params,t, t_mat_grid, Gamma_kalman_scale ,False)
                if neg_log_like == 1e12:
                    neg_log_like = np.nan
                results.append({
                    "parameter": param_name,
                    "value": val,
                    "neg_log_like": neg_log_like
                })

        results_df = pd.DataFrame(results)
        results_df.to_csv(f"./Empirical_Likelihoods/{firm}_likelihood_surface_CIR_transform.csv", index=False)



        #### plotting
        sns.set(style="whitegrid", font_scale=1.2)

        n_params = len(grids)
        n_cols = 3
        n_rows = math.ceil(n_params / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), sharex=False)
        axes = axes.flatten()  # flatten 2D array of axes

        for i, (param_name, grid_values) in enumerate(grids.items()):
            ax = axes[i]
            subset = results_df[results_df["parameter"] == param_name].copy()
            subset = subset.sort_values("value")
            subset["value"] = subset["value"].apply(float)

            # Plot likelihood curve
            sns.lineplot(data=subset, x="value", y="neg_log_like", ax=ax, color="royalblue", lw=2)

            # Add dashed line for true parameter value
            if param_name in kalman_params:
                # true_val = float(true_params[param_name])
                kalman_val =  float(kalman_params[param_name])
                # ax.axvline(true_val, color="red", linestyle="--", lw=1.5, alpha=0.8,
                #            label=f"True {param_name} = {round(true_val,4)}")
                ax.axvline(kalman_val, color="royalblue", linestyle="--", lw=1.5, alpha=0.8,
                        label=f"{param_name}")

            # Style adjustments
            ax.set_title(f"Likelihood profile: {param_name}", fontsize=13)
            ax.set_xlabel(param_name)
            ax.set_ylabel("Neg log-likelihood")
            ax.legend(frameon=False)
            ax.grid(True, alpha=0.3)

        # Hide any unused subplots (if number of parameters isn't multiple of 3)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(f"./Empirical_Likelihoods/{firm}_Likelihoods_CIR_transform.png", dpi=150)
        plt.close()






        # ============= INITIALIZE LHC ============= #
        # ### Simulate 1 LHC dataset with specific parameter Choises.
        for X_dim in [1,2,3]:
            directory = f"./Results/{firm}"

            filepath = os.path.join(directory, f"Kalman_resultsLHC_NX{X_dim}.npz")
            data = np.load(filepath)
            final_param = data["final_param"]
            lhc = LHC_single(0.00248,0.4,0.25)
            Y_dim,m = 1,X_dim
            # Here, parameters are set already
            rng = np.random.default_rng(4000)
            X0 = 0.4
            lhc.initialise_LHC(Y_dim,m,X0=X0,rng=rng)
            params = lhc.flatten_params()
            lhc.unflatten_params(final_param[:2*m+1])
            lhc_P = lhc.build_P_params(rng=rng)
            params_actual = [lhc.kappa, lhc.theta,lhc.gamma1,lhc.lambda_i,lhc.sigma, lhc.sigma_err]

            ### Run one kalman filter to see identification...
            # lhc_kalman_params,  Xn,Zn, Pn, se,ll= lhc.run_n_kalmans(T_path, t_mat_grid, CDS_simul_actual.T,base_seed=2000,n_restarts=1)
            # Read in the fitted kalman filter instead
            kalman_params = {
                "kappa": np.array(final_param[0:m]),
                "theta": np.array(final_param[m:2*m]),
                # "gamma1": np.array([lhc_kalman_params[2*m]]),
                "lambda_i": np.array(final_param[2*m+1:3*m+1]),
                "sigma": np.array(final_param[3*m+1:4*m+1]),
                "sigma_err": np.array([final_param[-1]])
            }

            # Spread (relative or absolute)
            grid_spread = {
                "kappa": 0.1,
                "theta": 0.1,
                # "gamma1": 0.05,
                "lambda_i": 0.3,
                "sigma": 0.1,
                "sigma_err": 0.1
            }

            n_points = 15
            results = []

            # =================== GRID FUNCTION =================== #
            def build_grid(center_params, label):
                grids = {}

                for key, val in center_params.items():
                    v = np.asarray(val)

                    spread = grid_spread[key]

                    # if key == "lambda_i":
                    #     up_b = (center_params["kappa"] - center_params["kappa"] * center_params["theta"] -
                    #             center_params["gamma1"])
                    #     low_b = (-center_params["kappa"] * center_params["theta"])
                    #     g = np.zeros((n_points,dim))
                    #     for i in range(dim):
                    #         g[:,i] = np.linspace(low_b[i], up_b[i], n_points)
                    # elif key == "theta":
                    #     lower = np.maximum(np.array([1e-6*X_dim]), v * (1 - spread))
                    #     upper = 1 - center_params["gamma1"] / center_params["kappa"]
                    #     dim = max(lower.shape[0],upper.shape[0])
                    #     g = np.zeros((n_points,dim))
                    #     for i in range(dim):
                    #         g[:,i] = np.linspace(lower[i], upper[i], n_points)
                    # elif key == "gamma1":
                    #     lower = max(1e-6, v * (1 - spread))
                    #     upper = center_params["kappa"] - center_params["kappa"] * center_params["theta"]
                    #     dim = upper.shape[0]
                    #     g = np.linspace(lower,  np.min(upper), n_points)
                    # elif key == "kappa":
                    #     lower = center_params["gamma1"] / (1 - center_params["theta"])
                    #     upper = v * (1 + spread)
                    #     dim = max(lower.shape[0],upper.shape[0])
                    #     g = np.zeros((n_points,dim))
                    #     for i in range(dim):
                    #         g[:,i] = np.linspace(lower[i], upper[i], n_points)

                    # elif key == "sigma":
                    #     sigma_constr = np.sqrt(-2*(center_params["gamma1"]  -  center_params["kappa"] +  center_params["kappa"] *  center_params["theta"] ))
                    #     sigma_thetakap = np.sqrt(2 * center_params["kappa"] * center_params["theta"])
                    #     upper = np.minimum(sigma_constr,sigma_thetakap)
                    #     dim = max(lower.shape[0],upper.shape[0])
                    #     g = np.zeros((n_points,dim))
                    #     for i in range(dim):
                    #         g[:,i] = np.linspace(0.05, upper[i], n_points)
                    # else:
                    #     lower = max(1e-6, v * (1 - spread))
                    #     upper = v * (1 + spread)
                    #     g = np.linspace(lower, upper, n_points)

                    if key == 'lambda_i':
                        lower = v - spread
                        max_bound = np.minimum( grid_spread['kappa'] -0.001,val + spread)
                        upper = max_bound
                    else:
                        lower = v * (1 - spread)
                        upper = v * (1 + spread)
                    dim = max(lower.shape[0],upper.shape[0])
                    g = np.zeros((n_points,dim))
                    for i in range(dim):
                        g[:,i] = np.linspace(lower[i], upper[i], n_points)
                    
                    g = np.linspace(lower, upper, n_points)

                    
                    grids[key] = g
                    

                
                print(f"\n✅ Built grid for {label}:")
                for k, v in grids.items():
                    print(f"{k}: {np.round(v, 4)}")

                return grids



            # =================== LIKELIHOOD LOOP =================== #

            for center_label, center_params in [("kalman", kalman_params)]:
            # for center_label, center_params in [("true", true_params)]:

                grids = build_grid(center_params, center_label)

                for pname, grid_values in grids.items():
                    print(f"--- Scanning {pname} around {center_label} center ---")
                    
                    # Loop over number of params
                    for index in range(grid_values[0,:].shape[0]):
                    # extract base parameter name and (optional) index
                        if grid_values[0,:].shape[0]>1:
                            base, idx = pname, index
                        else:
                            base, idx = pname, 0

                        for value in grid_values[:,index]:

                            pvals = copy.deepcopy(center_params)

                            if grid_values[0,:].shape[0] == 0:   # scalar
                                pvals[base][idx] = np.array([value])
                            else:             # vector --> update specific component
                                pvals[base][idx] = value

                            # unpack
                            kappa     = np.array(pvals["kappa"]).flatten()
                            theta     = np.array(pvals["theta"]).flatten()
                            # gamma1    = np.array(pvals["gamma1"]).flatten()
                            lambda_i  = np.array(pvals["lambda_i"]).flatten()
                            sigma     = np.array(pvals["sigma"]).flatten()
                            sigma_err = np.array(pvals["sigma_err"]).flatten()

                            # params_vec = np.concatenate([kappa, theta, gamma1, lambda_i, sigma, sigma_err])
                            params_vec = np.concatenate([kappa, theta, lambda_i, sigma, sigma_err])

                            # constraint check
                            constr = nonlinear_constraints(params_vec, m)
                            if np.any(constr < 0):
                                neg_log_like = np.nan
                            else:
                                # t_obs[::5],  T_M_grid[:,::5], CDS_obs[::5,:]
                                # neg_log_like = kalman_wrapper(
                                #     params_vec, T_path, T_path, t_mat_grid,
                                #     CDS_simul_actual.T,
                                #     X0=lhc.X0, m=lhc.m, r=lhc.r,
                                #     Y_dim=lhc.Y_dim, delta=lhc.delta, tenor=lhc.tenor
                                # )
                                neg_log_like = kalman_wrapper(
                                    params_vec, t, t, t_mat_grid,
                                    CDS_obs,
                                    X0=lhc.X0, m=lhc.m, r=lhc.r,
                                    Y_dim=lhc.Y_dim, delta=lhc.delta, tenor=lhc.tenor
                                )

                            results.append({
                                "center": center_label,
                                "parameter": pname,
                                "idx": idx,
                                "value": value,
                                "neg_log_like": neg_log_like
                            })


            # =================== SAVE + PLOT =================== #

            results_df = pd.DataFrame(results)
            results_df.to_csv(f"./Empirical_Likelihoods/{firm}_likelihood_surface_scan_dual_X{X_dim}.csv", index=False)
            sns.set(style="whitegrid", font_scale=1.2)

            # get all unique (param, idx) pairs
            param_idx_pairs = []
            for pname in sorted(results_df["parameter"].unique()):
                idxs = sorted(results_df.loc[results_df["parameter"]==pname, "idx"].unique())
                for idx in idxs:
                    param_idx_pairs.append((pname, idx))

            n_params = len(param_idx_pairs)
            n_cols = 4
            n_rows = math.ceil(n_params / n_cols)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
            axes = axes.flatten()

            for ax, (pname, idx) in zip(axes, param_idx_pairs):
                for center_label, color in zip(['kalman'], ["royalblue"]):
                    subset = results_df[(results_df["parameter"] == pname) &
                                        (results_df["idx"] == idx) &
                                        (results_df["center"] == center_label)].copy()
                    if subset.empty:
                        continue
                    subset = subset.sort_values("value").dropna()
                    sns.lineplot(data=subset, x="value", y="neg_log_like",
                                ax=ax, lw=2, label=f"{center_label}[{idx}]")

                # vertical line for true value
                # ax.axvline(true_params[pname][idx], color="blue", linestyle="--", lw=1.5)
                ax.axvline(kalman_params[pname][idx], color="royalblue", linestyle="--", lw=1.5)

                ax.set_title(f"Likelihood profile: {pname}_{idx}")
                ax.set_xlabel(f"{pname}_{idx}")
                ax.set_ylabel("Neg log-likelihood")
                ax.grid(True, alpha=0.3)

            # hide unused axes
            for j in range(len(param_idx_pairs), len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plt.savefig(f"./Empirical_Likelihoods/{firm}_Likelihoods_dual_X{X_dim}.png", dpi=150)
            plt.close()

