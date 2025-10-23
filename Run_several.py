import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from Models.LHCModels.LHC_single import LHC_single, rebuild_lhc_struct,kalmanfilter_opt,build_P_params
from Models.BaselineCIR_alternative.CIR_Multifactor import CIRIntensity
import os
if __name__ == "__main__":

    t_grid = [0 + 0.25* i for i in range(0, int(10 / 0.25)+1)]

    #### Preliminary investigation.
    sub_df = pd.read_excel("./Data/subset_data.xlsx")
    firms = ['BBVSM', 'BNP','CMZB','DANBNK','DB', 'HSBC', 'USPA']

    # firms = ['CMZB','DB', 'HSBC', 'USPA']
    # firms = ['DB', 'HSBC', 'USPA']
    # firms = ['DANBNK']
    # Pivot

#### SECTION 1: LHC KALMAN FITS
    # Loop over each firm in list.
    # for firm in firms:
    #     test_df = sub_df[(sub_df['Ticker']==firm)]
    #     test_df = test_df.pivot(index = ['Date','Ticker'],
    #                             columns='Tenor',values = 'Par Spread').reset_index()
    #     # Test on subset data ownly to get very few obs. One large spread increase to test.
    #     test_df['Years']= ((test_df['Date'] - test_df['Date'].min()).dt.total_seconds() / (365.25 * 24 * 3600)).drop_duplicates()

    #     t = np.array(test_df['Years'])

    #     mat_grid = np.array([1,3,5,7,10])
    #     t_mat_grid = np.ascontiguousarray(mat_grid[:, None] + t[None, :])   # shape (len(T_M_grid), len(t_obs))
    #     # Forward fill in case of nans.
    #     CDS_obs = np.array(test_df[['1Y','3Y','5Y','7Y','10Y']].ffill().bfill())

    #     ### Run models. For starters, we are just testing varois kalman versions.
    #     for X_dim in [3,2]:
    #         lhc = LHC_single( r=0.0252,delta=0.4,cds_tenor= 0.25 )
    #         lhc.initialise_LHC(Y_dim=1,X_dim=X_dim,X0=0.5,rng=None)


    #         optim_params,  Xn,Zn, Pn= lhc.run_n_kalmans(t, t_mat_grid, CDS_obs,base_seed = 300,n_restarts=1)
    #         Xn_kalman,Yn_kalman = lhc.kalman_X_Y(t,Xn)

    #         print(f'Optimal Paramerters {optim_params}')
    #         kappa, theta, gamma1 = optim_params[:lhc.m],optim_params[lhc.m:2*lhc.m], optim_params[2*lhc.m]

    #         default_intensity = lhc.default_intensity(Xn_kalman.T,Yn_kalman)
    #         directory = f"C:/Users/andre/OneDrive/KU, MAT-OEK/Kandidat/Thesis/Thesis_linear_CDS/Results/{firm}"
    #         filepath = os.path.join(directory, f"Kalman_resultsLHC_NX{X_dim}.npz")

    #         # Ensure directory exists
    #         os.makedirs(directory, exist_ok=True)

    #         np.savez(filepath,
    #                 final_param=optim_params,
    #                 Xn=Xn_kalman,
    #                 Yn=Yn_kalman,
    #                 Default_intensity = default_intensity,
    #                 CDS_model = Zn) #,
    #         print(f'Finised X_dim {X_dim}, firm {firm}')
        ### Then run comparison plots a la result_plots.
        # But run in seperate loop so no need for rerun.

##### SECTION 2: CIR MODEL FITS.
    # for firm in firms:

    #     r = 0.0252
    #     delta = 0.4
    #     tenor = 0.25

    #     test_df = sub_df[(sub_df['Ticker']==firm)]
    #     test_df = test_df.pivot(index = ['Date','Ticker'],
    #                             columns='Tenor',values = 'Par Spread').reset_index()
    #     # Test on subset data ownly to get very few obs. One large spread increase to test.
    #     test_df['Years']= ((test_df['Date'] - test_df['Date'].min()).dt.total_seconds() / (365.25 * 24 * 3600)).drop_duplicates()

    #     t = np.array(test_df['Years'])

    #     mat_grid = np.array([1,3,5,7,10])
    #     t_mat_grid = np.ascontiguousarray(mat_grid[:, None] + t[None, :])   # shape (len(T_M_grid), len(t_obs))
    #     # Forward fill in case of nans.
    #     CDS_obs = np.array(test_df[['1Y','3Y','5Y','7Y','10Y']].ffill().bfill())

    #     # Read in inferred survival probs.
    #     data = np.load(f"C:/Users/andre/OneDrive/KU, MAT-OEK/Kandidat/Thesis/Thesis_linear_CDS/Gamma_Calibration/{firm}/Data_{firm}.npz")
    #     t_mats_plots = data['t_mats_plots']
    #     survival=data['survival']
    #     Gamma = data['Gamma']
    #     default_prob = data['default_prob']

    #     t_mats_plots_kalman = t_mats_plots[np.isin(t_mats_plots,mat_grid).flatten()]

    #     survival_kalman = survival[:,np.isin(t_mats_plots, mat_grid).flatten()]
    #     Gamma_kalman = Gamma[:,np.isin(t_mats_plots, mat_grid).flatten()]


    #     # Negative process. Multiply by -1 everywhere. Let A and a get these too. 
    #     for X_dim in [2]:# , 2]:

    #         cir = CIRIntensity(r,delta,tenor,X_dim)
    #         x0 = np.array([0])
    #         params, Xn,Zn,Pn = cir.run_kalman_filter(t,t_mat_grid,Y=Gamma_kalman ,seed=2000)

    #         # Ttry with several restarts. 
    #         # params, Xn,Zn,Pn = cir.run_n_kalman(t,t_mat_grid,Y=Gamma_kalman,base_seed=206,n_restarts=5)

    #         # Set new optimal parameters too.
    #         cir.set_params(params)
    #         # default_prob_model= np.exp(-Zn)
    #         # Save values:

    #         # With Params in place, we can utilize CIR class to do pricing, simulations etc. 
    #         CDS_cir = np.zeros(survival_kalman.shape)
    #         for n in range(survival_kalman[:,0].shape[0]):    
    #             CDS_cir[n,:] = cir.cds_spread(Xn[n,:],params,t[n],t_mat_grid[:,n])
    #             print(f'Done with {(n+1)/survival_kalman[:,0].shape[0]} %')

    #         # Default intensity probability since inception (identical to state if dim=1)
    #         default_intensity = np.sum(Xn,axis=1)

    #         # Get induced default probability: 
    #         Yn = np.exp(-np.cumsum(default_intensity*(t[1]-t[0]))) # only approximates

    #         # Actual/observed

    #         np.savez(f"C:/Users/andre/OneDrive/KU, MAT-OEK/Kandidat/Thesis/Thesis_linear_CDS/Results/{firm}/Kalman_resultsCIR_Xdim{X_dim}.npz",
    #                 final_param=params,
    #                 Xn=Xn,
    #                 Zn=Zn,
    #                 Pn = Pn,
    #                 Yn = Yn,
    #                 default_intensity = default_intensity,
    #                 CDS_cir = CDS_cir)




##### Plots for comparison.

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    from Models.Utils import global_fit_measures
    from Models.LHCModels.LHC_single import LHC_single
    from Models.BaselineCIR_alternative.CIR_numba import calc_cds
    from Models.LHCModels.LHC_single import cds_fun
    from Models.BaselineCIR_alternative.CIR_Multifactor import CIRIntensity
    import matplotlib.pyplot as plt
    from Result_plots import print_model_params

    for firm in firms:
        test_df = sub_df[(sub_df['Ticker']==firm)]
        test_df = test_df.pivot(index = ['Date','Ticker'],
                                columns='Tenor',values = 'Par Spread').reset_index()
        # Test on subset data ownly to get very few obs. One large spread increase to test.
        test_df['Years']= ((test_df['Date'] - test_df['Date'].min()).dt.total_seconds() / (365.25 * 24 * 3600)).drop_duplicates()

        t = np.array(test_df['Years'])

        mat_grid = np.array([1,3,5,7,10])
        t_mat_grid = np.ascontiguousarray(mat_grid[:, None] + t[None, :])   # shape (len(T_M_grid), len(t_obs))

        # Forwrard fill again. Back fill in case any initial missing
        CDS_obs = np.array(test_df[['1Y','3Y','5Y','7Y','10Y']].ffill().bfill())

        directory = f"C:/Users/andre/OneDrive/KU, MAT-OEK/Kandidat/Thesis/Thesis_linear_CDS/Results/{firm}"
        filepath = os.path.join(directory, f"Kalman_resultsLHC_NX{2}.npz")
        data_2 = np.load(filepath)
        final_paramLHCK = data_2["final_param"]
        XnLHCK = data_2["Xn"]
        YnLHCK=data_2["Yn"]
        ZnLHCK = data_2["CDS_model"]
        Default_intensityLHCK = data_2["Default_intensity"]

        # Example usage:
        print_model_params("LHC Kalman", final_paramLHCK, m=2)

        filepath = os.path.join(directory, f"Kalman_resultsLHC_NX{3}.npz")
        data = np.load(filepath)
        final_paramLHCK3 = data["final_param"]
        XnLHCK3 = data["Xn"]
        YnLHCK3=data["Yn"]
        ZnLHCK3 = data["CDS_model"]
        Default_intensityLHCK3 = data["Default_intensity"]
        print_model_params("LHC Kalman", final_paramLHCK, m=3)

        filepath = os.path.join(directory, f"Kalman_resultsCIR_Xdim{2}.npz")
        data = np.load(filepath)
        final_paramCIR2=data['final_param']
        XnCIR2=data['Xn']
        ZnCIR2=data['Zn']
        PnCIR2 = data['Pn']
        YnCIR2 = data['Yn']
        default_intensityCIR2 = data['default_intensity']
        CDS_cirCIR2 = data['CDS_cir']

        # Example usage:
        print_model_params("CIR", final_paramCIR2, m=2)

        #### Test of CIR comp

        filepath = os.path.join(directory, f"Kalman_resultsCIR_Xdim{3}.npz")
        data = np.load(filepath)
        final_paramCIR3=data['final_param']
        XnCIR3=data['Xn']
        ZnCIR3=data['Zn']
        PnCIR3 = data['Pn']
        YnCIR3 = data['Yn']
        default_intensityCIR3 = data['default_intensity']
        CDS_cirCIR3 = data['CDS_cir']
        print_model_params("CIR", final_paramCIR3, m=3)

        # Loop through maturities and make a separate plot for each
        for i in range(XnLHCK3.shape[1]):
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(t, XnLHCK3[:,i], "-", alpha=0.7, color='blue', label="LHC Kalman, m=3")
            ax.plot(t, XnCIR3[:,i], "-", alpha=0.7, color='green', label="CIR Kalman, m=3")
            if i< 2:
                ax.plot(t, XnLHCK[:,i], "-", alpha=0.7, color='orange', label="LHC Kalman, m=2")
                ax.plot(t, XnCIR2[:,i], "-", alpha=0.7, color='red', label="CIR Kalman, m=2")

            ax.grid(True)
            ax.set_xlabel("Time (years)")
            ax.set_ylabel("Model States")
            ax.set_title(f"Model States X{i+1} maturity ({firm})")
            ax.legend()

            fig.tight_layout()
            save_file = os.path.join(directory, f"States_X{i+1}.png")
            fig.savefig(save_file, dpi=150)
            plt.close(fig)


        # Default intensities
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(t, Default_intensityLHCK , "-", alpha=0.7, color='orange', label=f"LHC Kalman,m=2")
        ax.plot(t, Default_intensityLHCK3 , "-", alpha=0.7, color='blue', label=f"LHC Kalman,m=3")
        ax.plot(t, default_intensityCIR2 , "-", alpha=0.7, color='red', label=f"CIR Kalman,m=2")
        ax.plot(t, default_intensityCIR3 , "-", alpha=0.7, color='green', label=f"CIR Kalman,m=3")



        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Default Intensity")
        ax.set_title(f"Default intensities in different models, {firm}")
        ax.legend()
        ax.grid()
        fig.tight_layout()
        fig.savefig(os.path.join(directory, "DefaultIntensities.png"), dpi=150)
        plt.close(fig)


        # Recreated Spreads

        # Example list of maturities (adjust names if your columns differ)
        maturities = [1, 3, 5, 7, 10]  # or ['1Y','3Y','5Y','7Y','10Y']

        # Loop through maturities and make a separate plot for each
        for m,mat in enumerate(maturities):
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(t, ZnLHCK[:,m], "-", alpha=0.7, color='orange', label="LHC Kalman, m=2")
            ax.plot(t, ZnLHCK3[:,m], "-", alpha=0.7, color='blue', label="LHC Kalman,m=3")
            ax.plot(t, CDS_cirCIR2[:,m], "-", alpha=0.7, color='red', label="CIR Kalman, m=2")
            ax.plot(t, CDS_cirCIR3[:,m], "-", alpha=0.7, color='green', label="CIR Kalman,m=3")
            ax.plot(t, CDS_obs[:,m], "o", alpha=0.5, color='black', label="Observations")

            ax.grid(True)
            ax.set_xlabel("Time (years)")
            ax.set_ylabel("Model Spreads / Intensity")
            ax.set_title(f"Model Spreads at {maturities[m]}-year maturity {firm}")
            ax.legend()

            fig.tight_layout()
            save_file = os.path.join(directory, f"Spreads_{maturities[m]}Y.png")
            fig.savefig(save_file, dpi=150)
            plt.close(fig)


        # Survival process.
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(t, YnLHCK , "-", alpha=0.7, color='orange', label=f"LHC Kalman,m=2")
        ax.plot(t, YnLHCK3 , "-", alpha=0.7, color='blue', label=f"LHC Kalman,m=3")
        ax.plot(t, YnCIR2 , "-", alpha=0.7, color='red', label=f"CIR Kalman,m=2")
        ax.plot(t, YnCIR3 , "-", alpha=0.7, color='green', label=f"CIR Kalman,m=3")


        ax.grid()
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Survival prob Intensity")
        ax.set_title(f"Survival probabilities in different models, {firm}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(directory, "SurvivalProb.png"), dpi=150)
        plt.close(fig)



        ### Compute Global measures of fit.
        # Stack together CDS frames

        models = [ZnLHCK,ZnLHCK3,CDS_cirCIR2,CDS_cirCIR3] # stacked fitted CDS spreads.

        gfm = global_fit_measures(CDS_obs, models)

        rmse_series, rmse = gfm.rmse()
        ape_series, ape = gfm.ape()
        aae_series, aae = gfm.aae()
        arpe_series, arpe = gfm.arpe()

        # Example structure:
        cols_names = ["LHC Kalman,m=2", "LHC Kalman,m=3","CIR Kalman,m=2", "CIR Kalman,m=3"]

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
        fig.savefig(os.path.join(directory, "Global_fit_errors.png"), dpi=150)
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

