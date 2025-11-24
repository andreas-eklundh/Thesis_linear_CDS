
##### Plots for comparison. TODO: Move to other file.
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from Models.LHCModels.LHC_single import LHC_single, rebuild_lhc_struct,kalmanfilter_opt,build_P_params
from Models.BaselineCIR_alternative.CIR_Multifactor import CIRIntensity
import os
full_global_summary = []
if __name__ == "__main__":

    t_grid = [0 + 0.25* i for i in range(0, int(10 / 0.25)+1)]

    #### Preliminary investigation.
    sub_df = pd.read_excel("./Data/subset_data.xlsx")
    firms = ['CMZB','DANBNK','MONTE', 'SVSKHB'] # IG,IG,HY,IG
    firms = [ 'CMZB'] # IG,IG,HY,IG

    # Pivot

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

    # X_dims = [1,2,3]
    X_dims = [2,3]
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
        for i in X_dims:
            filepath = os.path.join(directory, f"Kalman_resultsLHC_NX{i}.npz")
            data = np.load(filepath)
            final_paramLHCK = data["final_param"]
            XnLHCK = data["Xn"]
            YnLHCK=data["Yn"]
            PnLHC = data["Pn"]
            ZnLHCK = data["CDS_model"]
            Default_intensityLHCK = data["Default_intensity"]
            print_model_params("LHC Kalman", final_paramLHCK, m=i)

            filepath = os.path.join(directory, f"Filipovic_LHC_NX{i}.npz")
            data = np.load(filepath)
            final_paramLHC =data["final_param"]
            XnLHC=data["Xn"]
            YnLHC=data["Yn"]
            Default_intensityLHC = data["Default_intensity"]
            CDS_LHC = data["CDS_model"]
            print_model_params("LHC Filipovic", final_paramLHC, m=i)


            filepath = os.path.join(directory, f"Kalman_resultsCIR_Xdim{i}.npz")
            data = np.load(filepath)
            final_paramCIR=data['final_param']
            XnCIR=data['Xn']
            ZnCIR=data['Zn']
            PnCIR = data['Pn']
            YnCIR = data['Yn']
            default_intensityCIR = data['default_intensity']
            CDS_cirCIR = data['CDS_cir']

            # # Example usage:
            print_model_params("CIR", final_paramCIR, m=i)

            # Loop through maturities and make a separate plot for each
            for j in range(XnLHC.T.shape[1]):
                fig, ax = plt.subplots(figsize=(10, 6))

                ax.plot(t, XnLHCK[:,j], "-", alpha=0.7, color='blue', label=f"LHC Kalman")
                ax.plot(t, XnLHC.T[:,j], "-", alpha=0.7, color='red', label=f"LHC Filipovic")
                ax.plot(t, XnCIR[:,j], "-", alpha=0.7, color='green', label=f"CIR Kalman")

                ax.grid(True)
                ax.set_xlabel("Time (years)")
                ax.set_ylabel("Model States")
                ax.set_title(f"Model States X{j+1} maturity ({firm})")
                ax.legend()

                fig.tight_layout()
                save_file = os.path.join(directory, f"States_Xdim{i}_X{j+1}.png")
                fig.savefig(save_file, dpi=150)
                plt.close(fig)


            # Default intensities
            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(t, Default_intensityLHCK , "-", alpha=0.7, color='blue', label=f"LHC Kalman")
            ax.plot(t, Default_intensityLHC , "-", alpha=0.7, color='red', label=f"LHC Filipovic")
            ax.plot(t, default_intensityCIR , "-", alpha=0.7, color='green', label=f"CIR Kalman")



            ax.set_xlabel("Time (years)")
            ax.set_ylabel("Default Intensity")
            ax.set_title(f"Default intensities in different models, {firm}")
            ax.legend()
            ax.grid()
            fig.tight_layout()
            fig.savefig(os.path.join(directory, f"DefaultIntensities_Xdim{i}.png"), dpi=150)
            plt.close(fig)


            # Recreated Spreads

            # Example list of maturities (adjust names if your columns differ)
            maturities = [1, 3, 5, 7, 10]  # or ['1Y','3Y','5Y','7Y','10Y']

            # Loop through maturities and make a separate plot for each
            for m,mat in enumerate(maturities):
                fig, ax = plt.subplots(figsize=(10, 6))

                ax.plot(t, ZnLHCK[:,m], "-", alpha=0.7, color='blue', label="LHC Kalman")
                ax.plot(t, CDS_LHC[:,m], "-", alpha=0.7, color='red', label="LHC Filipovic")
                ax.plot(t, CDS_cirCIR[:,m], "-", alpha=0.7, color='green', label="CIR Kalman")
                ax.plot(t, CDS_obs[:,m], "o", alpha=0.5, color='black', label="Observations")

                ax.grid(True)
                ax.set_xlabel("Time (years)")
                ax.set_ylabel("Model Spreads / Intensity")
                ax.set_title(f"Model Spreads at {maturities[m]}-year maturity {firm}")
                ax.legend()

                fig.tight_layout()
                save_file = os.path.join(directory, f"Spreads_Xdim{i}_{maturities[m]}Y.png")
                fig.savefig(save_file, dpi=150)
                plt.close(fig)


            # Survival process.
            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(t, YnLHCK , "-", alpha=0.7, color='blue', label=f"LHC Kalman")
            ax.plot(t, YnLHC, "-", alpha=0.7, color='red', label=f"LHC Filipovic")
            ax.plot(t, YnCIR , "-", alpha=0.7, color='green', label=f"CIR Kalman")


            ax.grid()
            ax.set_xlabel("Time (years)")
            ax.set_ylabel("Survival prob Intensity")
            ax.set_title(f"Survival probabilities in different models, {firm}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(directory, f"SurvivalProb_Xdim{i}.png"), dpi=150)
            plt.close(fig)



            ### Compute Global measures of fit.
            # Stack together CDS frames

            models = [ZnLHCK,CDS_LHC,CDS_cirCIR] # stacked fitted CDS spreads.
            # models = [CDS_LHC,CDS_cirCIR] # stacked fitted CDS spreads.

            gfm = global_fit_measures(CDS_obs, models)

            rmse_series, rmse = gfm.rmse()
            ape_series, ape = gfm.ape()
            aae_series, aae = gfm.aae()
            arpe_series, arpe = gfm.arpe()

            # Example structure:
            cols_names = [f"LHC Kalman,m={i}", f"LHC Filipovic,m={i}",f"CIR Kalman,m={i}"]
            # cols_names = [f"LHC Filipovic,m={i}",f"CIR Kalman,m={i}"]

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
            colors = {cols_names[0]:'blue',cols_names[1]:'red',cols_names[2]:'green'}
            # colors = {cols_names[0]:'red',cols_names[1]:'green'}

            for ax, (label, df) in zip(axes, metrics):
                for col in df.columns:
                    ax.plot(test_df['Date'], df[col], label=col, lw=1.5,color=colors[col])
                ax.set_title(label, fontsize=12, fontweight="bold")
                ax.set_xlabel("Observation")
                ax.set_ylabel(label)
                ax.grid(True, alpha=0.3)

            # Add legend (outside for clarity)
            axes[-1].legend(title="Models", loc="upper right", bbox_to_anchor=(1.25, 1.0))
            
            fig.suptitle("Risk Measure Time Series by Model", fontsize=14, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(os.path.join(directory, f"Global_fit_errors_Xdim{i}.png"), dpi=150)
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

            full_global_summary.append(global_summary)


    stopper = 1