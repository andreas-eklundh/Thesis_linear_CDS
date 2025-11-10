############## INVESTIGATION OF LIKELIHOODS ##############
#### Purpose: investigate how the likelihood region looks as a function of the parameters.
import numpy as np
from Models.LHCModels.LHC_single import LHC_single, kalman_wrapper,build_P_params,rebuild_lhc_struct
import pandas as pd
###### Create grids.
from Models.LHCModels.LHC_single import rebuild_lhc_struct, get_CDS_Model, nonlinear_constraints
from Models.BaselineCIR_alternative.CIR_Multifactor import CIRIntensity as CIR
from Models.BaselineCIR_alternative.Gamma_solver import DeterministicGamma as Gamma_class
from scipy.stats import norm
import math
import seaborn as sns
import matplotlib.pyplot as plt

# ============= INITIALIZE CIR ============= #
# ### Simulate 1 LHC dataset with specific parameter Choises.
if __name__ == '__main__':
    X_dim =1
    T,M = 10, 1000
    cir = CIR(0.0252,0.4,0.25,X_dim)
    # Here, parameters are set already
    rng = np.random.default_rng(2000)
    seed = 2000
    r = cir.r
    delta = cir.delta
    tenor = cir.tenor
    cir.set_params(params=None, seed=seed)
    print(cir.kappa,cir.theta,cir.sigma,cir.lambda1,cir.sigma_err)
    params_cir = np.concatenate([cir.kappa,cir.theta,cir.sigma,cir.lambda1,cir.sigma_err])
    # Simulate. We are using an Euler discretization.
    # Set initial lambda to the one we would get in a LHC model.
    # initial values - just decide on Q
    kappa_p = cir.kappa - cir.lambda1
    theta_p = cir.kappa * cir.theta / kappa_p
    alpha = 2 * kappa_p*  theta_p /  cir.sigma**2
    beta = 2 * kappa_p /  cir.sigma**2
    # CIR values.
    lambda0 = alpha / beta
    # This is undekkr Q
    T_return,lambda_eul_Q = cir.simulate_intensity(lambda0=lambda0,T=T,M=M,scheme="Euler",seed=seed,measure='P')
    T_return,lambda_mil_Q = cir.simulate_intensity(lambda0=lambda0,T=T,M=M,scheme="Milstein",seed=seed,measure='P')
    simuls = np.hstack([lambda_eul_Q,lambda_mil_Q])
    plt.plot(T_return,lambda_eul_Q,color = 'red', label = 'euler' )
    plt.plot(T_return,lambda_mil_Q,color = 'blue', label = 'milstein' )
    plt.legend()
    plt.show()
    
    mat_grid = np.array([1,3,5,7,10])
    t_mat_grid = np.ascontiguousarray(mat_grid[:, None] + T_return[None, :])   # shape (len(T_M_grid), len(t_obs))


    #### plotting

    #### In the approach that one would actually follow:

    # CDS_cir = np.ones((t_mat_grid.T.shape))
    # for i in range(t_mat_grid.shape[1]):
    #     lambda_curr  = np.array([lambda_mil_Q[i]])
    #     mat_curr = t_mat_grid[:,i]
    #     CDS_cir[i,:] = cir.cds_spread(lambda_curr,params_cir,T_return[i],mat_curr)

    # Add noise to spreads (as that is what we somewhat assume)


    # model = Gamma_class(r, delta, tenor)
    # t_mats = np.concatenate(([0],mat_grid))
    # extrapolate_grid = np.array([i  for i in range(int(np.max(mat_grid))+ 1)])

    # Gamma = np.zeros((CDS_cir.shape[0], extrapolate_grid.shape[0]))
    # cali_params = np.zeros((CDS_cir.shape[0], mat_grid.shape[0]))

    # for t_idx in range(CDS_cir.shape[0]):
    #     # Calibrate back hazard rates
    #     t_grid_payments = np.array([tenor*i for i in range(int(np.max(mat_grid)/tenor)+1)])
    #     cali_params[t_idx, : ] = model.calibrate_deterministic(CDS_cir[t_idx,:] , mat_grid, 0.0, t_grid_payments)
    #     # Generate the survial probabilities/survival process
    #     for i in range(extrapolate_grid.shape[0]):
    #         Gamma[t_idx,i] = model.Gamma_fun(cali_params[t_idx, : ],extrapolate_grid[i],t_mats)
        
    #     survival= np.exp(-Gamma )


    # Gamma_kalman = Gamma[:,np.isin(extrapolate_grid, mat_grid).flatten()]

    #Select only params at maturity.

    #Add noise to kalman
    # R = norm.rvs(size = (Gamma_kalman.shape[0]*Gamma_kalman.shape[1]),
    #               scale = cir.sigma_err,random_state=10).reshape(CDS_cir.shape) # simulate at beginning - faster!
    # Gamma_kalman_noise = Gamma_kalman + R

    # While the above is the actual approach taken when real data, test completely in 
    # a test environment. 
    Lambda_kalman = np.zeros(shape = (T_return.shape[0], mat_grid.shape[0]))
    for i in range(T_return.shape[0]):
        Lambda_kalman[i,:] =  - np.log(cir.Laplace_Transform(params_cir,lambda_mil_Q[i],mat_grid))
    R = norm.rvs(size = (Lambda_kalman.shape[0]*Lambda_kalman.shape[1]),
                 scale = cir.sigma_err,random_state=10).reshape(Lambda_kalman.shape) # simulate at beginning - faster!

    Lambda_kalman_noise = Lambda_kalman + R
    ### Run one kalman filter to see identification...
    # params_cir_kalman_trans ,Xn_trans,Zn,Pn = cir.run_kalman_filter(T_return, t_mat_grid, Gamma_kalman_noise)
    params_cir_kalman , Xn,Zn,Pn,se = cir.run_kalman_filter(T_return, t_mat_grid, Lambda_kalman_noise)
    # Recreate Kalman spreads
    CDS_cir_kalman = np.zeros(Lambda_kalman_noise.shape)
    CDS_cir_kalman_trans = np.zeros(Lambda_kalman_noise.shape)
    # for n in range(Lambda_kalman_noise[:,0].shape[0]):
    #     CDS_cir_kalman[n,:] = cir.cds_spread(Xn[n,:],params_cir_kalman,T_return[n],t_mat_grid[:,n])
    #     CDS_cir_kalman_trans[n,:] = cir.cds_spread(Xn_trans[n,:],params_cir_kalman_trans,T_return[n],t_mat_grid[:,n])
    #     print(f'Done with {(n+1)/Lambda_kalman_noise[:,0].shape[0]} %')

    # params_cir_kalman , Xn,Zn,Pn = cir.run_kalman_filter(T_return, t_mat_grid, Gamma_kalman_noise)


    # assuming lhc.flatten_params() returns [kappa, theta, gamma1, sigma, lambda_i, sigma_err]
    true_params = {
        "kappa": params_cir[0],
        "theta": params_cir[1],
        "sigma": params_cir[2],
        "lambda1": params_cir[3],
        "sigma_err": params_cir[4]
    }

    kalman_params = {
        "kappa": params_cir_kalman[0],
        "theta": params_cir_kalman[1],
        "sigma": params_cir_kalman[2],
        "lambda1": params_cir_kalman[3],
        "sigma_err": params_cir_kalman[4]
    }


    # Define relative/absolute spreads for scanning
    grid_spread = {
        "kappa": 0.3,
        "theta": 0.3,
        "sigma": 0.3,
        "lambda1": 0.5,     # absolute
        "sigma_err": 0.3
    }

    n_points = 60
    grids = {}
    for key, val in true_params.items():
        spread = grid_spread[key]
        val = float(val)  # ensure scalar
        if key == "lambda1":
            # min_bound = - cir.theta*cir.kappa
            max_bound = cir.kappa
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
            pvals = true_params.copy()
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
            if np.any(cir.feller_constraint(params_cir) <0):
                
                print('Feller Condition Failed')
            neg_log_like = cir.Kalman(params,T_return, t_mat_grid, Lambda_kalman_noise ,False)
            if neg_log_like == 1e12:
                neg_log_like = np.nan
            results.append({
                "parameter": param_name,
                "value": val,
                "neg_log_like": neg_log_like
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv("./Likelihoods/likelihood_surface_CIR_transform.csv", index=False)



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
        if param_name in true_params:
            true_val = float(true_params[param_name])
            kalman_val =  float(kalman_params[param_name])
            ax.axvline(true_val, color="red", linestyle="--", lw=1.5, alpha=0.8,
                       label=f"True {param_name} = {round(true_val,4)}")
            ax.axvline(kalman_val, color="green", linestyle="--", lw=1.5, alpha=0.8,
                       label=f"Kalman {param_name} = {round(kalman_val,4)}")

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
    plt.savefig("./Likelihoods/Likelihoods_CIR_transform.png", dpi=150)
    plt.close()






    # ============= INITIALIZE LHC ============= #
    # ### Simulate 1 LHC dataset with specific parameter Choises.
    lhc = LHC_single(0.0252,0.4,0.25)
    Y_dim,m = 1,1
    # Here, parameters are set already
    rng = np.random.default_rng(25524)
    X0 = 0.4
    lhc.initialise_LHC(Y_dim,m,X0=X0,rng=rng)
    params = lhc.flatten_params()
    lhc.unflatten_params(params[:2*m+1])
    lhc_P = lhc.build_P_params(rng=rng)
    params_actual = [lhc.kappa, lhc.theta,lhc.gamma1,lhc.lambda_i,lhc.sigma, lhc.sigma_err]

    # Set initial values as in kalman filter.
    mu1 = lhc.solve_mu1(lhc.kappa,lhc.theta,lhc.gamma1,lhc.lambda_i)
    chi0 = lhc.compute_stationary(lhc.kappa,lhc.theta,lhc.m,lhc.gamma1,mu1,lhc.lambda_i) # np.array([1] + [X0]*m)
    chi0 = np.append([1],chi0)

    T,M = 10, 300
    mat_grid = np.array([1,3,5,7,10]) # Typical maturity grid

    n_mat = mat_grid.shape[0]
    # Set initial values to be mu.
    # Use the Euler scheme as estimation is happening under too. 
    T_path, chi_Q = lhc.simul_latent_states(chi0=chi0,T=T,M=M,n_mat=n_mat,seed=3000,scheme='Milstein',
                                            measure='P')


    # Holld maturity to be 5
    t_mat_grid = np.ascontiguousarray(mat_grid[:, None] + T_path[None, :])   # shape (len(T_M_grid), len(t_obs))

    t0 = T_path

    lhc_numba = rebuild_lhc_struct(lhc.kappa, lhc.theta, lhc.gamma1[0],
                                lhc.r, lhc.Y_dim, lhc.delta, lhc.tenor)


    # Draw noise vector:
    R = norm.rvs(size = (t_mat_grid.shape[0]*t_mat_grid.shape[1]),scale = lhc.sigma_err).reshape(t_mat_grid.shape) # simulate at beginning - faster!
    CDS_simul_actual = get_CDS_Model(T_path, t0, t_mat_grid, chi_Q.T, lhc_numba) + R
    # One could try without noise!


    ### Run one kalman filter to see identification...
    lhc_kalman_params,  Xn,Zn, Pn, se= lhc.run_n_kalmans(T_path, t_mat_grid, CDS_simul_actual.T,base_seed=2000,n_restarts=1)
    print(lhc_kalman_params)
    # =================== PARAM SETUP =================== #

    true_params = {
        "kappa": params_actual[0][0],
        "theta": params_actual[1][0],
        "gamma1": params_actual[2][0],
        "lambda_i": params_actual[3][0],
        "sigma": params_actual[4][0],
        "sigma_err": params_actual[5][0]
    }

    kalman_params = {
        "kappa": lhc_kalman_params[0],
        "theta": lhc_kalman_params[1],
        "gamma1": lhc_kalman_params[2],
        "lambda_i": lhc_kalman_params[3],
        "sigma": lhc_kalman_params[4],
        "sigma_err": lhc_kalman_params[5]
    }

    # Spread (relative or absolute)
    grid_spread = {
        "kappa": 0.2,
        "theta": 0.2,
        "gamma1": 0.2,
        "lambda_i": 0.9,
        "sigma": 0.2,
        "sigma_err": 0.5
    }

    n_points = 15
    results = []

    # =================== GRID FUNCTION =================== #

    def build_grid(center_params, label):
        """
        Build parameter grids centered around 'center_params' (true or kalman).
        Returns dict of 1D arrays.
        """
        grids = {}
        for key, val in center_params.items():
            spread = grid_spread[key]

            if key == "lambda_i":
                up_b = (center_params["kappa"] - center_params["kappa"] * center_params["theta"] -
                        center_params["gamma1"])
                # lowp_b = (-center_params["kappa"] * center_params["theta"])
                grids[key] = np.linspace(-1, up_b , n_points)

            elif key == "theta":
                lower = max(1e-6, val * (1 - spread))
                upper = 1 - center_params["gamma1"] / center_params["kappa"]
                grids[key] = np.linspace(lower, upper, n_points)

            elif key == "gamma1":
                lower = max(1e-6, val * (1 - spread))
                upper = center_params["kappa"] - center_params["kappa"] * center_params["theta"]
                # upper =  center_params["theta"]
                grids[key] = np.linspace(lower, upper, n_points)

            elif key == "kappa":
                lower = center_params["gamma1"] / (1 - center_params["theta"])
                upper = val * (1 + spread)
                grids[key] = np.linspace(lower, upper, n_points)

            elif key == "sigma":
                sigma_thetakap = np.sqrt(2 * (
                    center_params["kappa"]-center_params["lambda_i"] -
                    center_params["kappa"]*center_params["theta"]-center_params["gamma1"]))
            
                grids[key] = np.linspace(0.05, sigma_thetakap, n_points)

            else:
                lower = max(1e-6, val * (1 - spread))
                upper = val * (1 + spread)
                grids[key] = np.linspace(lower, upper, n_points)

            # clip to avoid negatives
            if key != 'lambda_i':
                grids[key] = np.clip(grids[key], 1e-6, None)

        print(f"\n✅ Built grid for {label}:")
        for k, v in grids.items():
            print(f"{k}: {np.round(v, 4)}")

        return grids


    # =================== LIKELIHOOD LOOP =================== #

    for center_label, center_params in [("true", true_params), ("kalman", kalman_params)]:
        grids = build_grid(center_params, center_label)

        for param_name, grid_values in grids.items():
            print(f"--- Scanning {param_name} around {center_label} center ---")

            for value in grid_values:
                # build a working copy of parameters
                pvals = center_params.copy()
                pvals[param_name] = value

                # unpack
                kappa = np.array(pvals["kappa"]).flatten()
                theta = np.array(pvals["theta"]).flatten()
                gamma1 = np.array(pvals["gamma1"]).flatten()
                sigma = np.array(pvals["sigma"]).flatten()
                lambda_i = np.array(pvals["lambda_i"]).flatten()
                sigma_err = np.array(pvals["sigma_err"]).flatten()

                # full param vector
                params_vec = np.concatenate([
                    kappa, theta, gamma1, lambda_i, sigma, sigma_err
                ])

                # constraint check
                constr = nonlinear_constraints(params_vec, m)
                if np.any(constr < 0):
                    neg_log_like = np.nan
                else:
                    neg_log_like = kalman_wrapper(
                        params_vec, T_path, T_path, t_mat_grid,
                        CDS_simul_actual.T,
                        X0=lhc.X0, m=lhc.m, r=lhc.r,
                        Y_dim=lhc.Y_dim, delta=lhc.delta, tenor=lhc.tenor
                    )

                results.append({
                    "center": center_label,
                    "parameter": param_name,
                    "value": value,
                    "neg_log_like": neg_log_like
                })


    # =================== SAVE + PLOT =================== #

    results_df = pd.DataFrame(results)
    results_df.to_csv("./Likelihoods/likelihood_surface_scan_dual.csv", index=False)

    sns.set(style="whitegrid", font_scale=1.2)

    n_params = len(true_params)
    n_cols = 3
    n_rows = math.ceil(n_params / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), sharex=False)
    axes = axes.flatten()

    for i, param_name in enumerate(true_params.keys()):
        ax = axes[i]

        for center_label, color in zip(["kalman", "true"], ["royalblue", "orange"]):
            subset = results_df[
                (results_df["parameter"] == param_name) &
                (results_df["center"] == center_label)
            ].copy()
            subset = subset.sort_values("value")
            subset["value"] = pd.to_numeric(subset["value"], errors="coerce")
            subset["neg_log_like"] = pd.to_numeric(subset["neg_log_like"], errors="coerce")
            subset = subset.dropna(subset=["value", "neg_log_like"])
            sns.lineplot(
                data=subset, x="value", y="neg_log_like",
                ax=ax, lw=2, label=f"{center_label.capitalize()} center", color=color
            )

        # Vertical lines
        true_val = float(true_params[param_name])
        kalman_val = float(kalman_params[param_name])
        ax.axvline(true_val, color="red", linestyle="--", lw=1.5, label=f"True = {round(true_val,4)}")
        ax.axvline(kalman_val, color="green", linestyle="--", lw=1.5, label=f"Est = {round(kalman_val,4)}")

        # Style
        ax.set_title(f"Likelihood profile: {param_name}", fontsize=13)
        ax.set_xlabel(param_name)
        ax.set_ylabel("Neg log-likelihood")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)

    # hide unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig("./Likelihoods/Likelihoods_dual.png", dpi=150)
    plt.close()

    test = 1