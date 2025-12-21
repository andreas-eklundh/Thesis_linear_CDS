
#### Plots for comparison. TODO: Move to other file.
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from Models.LHCModels.LHC_single import LHC_single,nonlinear_constraints, rebuild_lhc_struct,kalmanfilter_opt,build_P_params
from Models.LHCModels.LHC_wGamma1 import LHC_single as LHC_wGamma1
from Models.LHCModels.LHC_wGamma1 import nonlinear_constraints as nonlinear_constraints_wgamma

from Models.BaselineCIR_alternative.CIR_Multifactor import CIRIntensity
import os
if __name__ == "__main__":

    t_grid = [0 + 0.25* i for i in range(0, int(10 / 0.25)+1)]

    ### Preliminary investigation.
    sub_df = pd.read_excel("./Data/subset_data.xlsx")
    firms = ['CMZB','DANBNK','MONTE', 'SVSKHB'] # IG,IG,HY,IG
    firms = ['DANBNK','MONTE'] # IG,IG,HY,IG
    
    full_global_summary = {'CMZB': {},'DANBNK': {},
                           'MONTE': {}, 'SVSKHB': {}}

    # Pivot

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    from Models.Utils import global_fit_measures
    from Models.LHCModels.LHC_single import LHC_single
    
    from Models.LHCModels.LHC_single import cds_fun
    from Models.BaselineCIR_alternative.CIR_Multifactor import CIRIntensity
    import matplotlib.pyplot as plt

    ### printing results params
    def print_model_params(name, final_param, m):
        """
        Pretty-print model parameters assuming structure:
        [kappa (m), theta (m), gamma1 (1), kappa_p (m), theta_p (m), sigma (m), sigma_err (1)]
        """
        print(f"\n{'='*60}")
        print(f"Model: {name}")
        print(f"{'='*60}")
        if name == 'LHC Kalman':
            idx = 0
            kappa = final_param[idx:idx + m]; idx += m
            theta = final_param[idx:idx + m]; idx += m
            gamma1 = final_param[idx]; idx += 1
            lambda_i = final_param[idx:idx + m]; idx += m
            sigma = final_param[idx:idx + m]; idx += m
            sigma_err = final_param[idx] if idx < len(final_param) else np.nan

            print("κ      =", np.round(kappa, 4))
            print("θ      =", np.round(theta, 4))
            print("γ₁     =", np.round(gamma1, 4))
            print("lambda     =", np.round(lambda_i, 4))
            print("σ      =", np.round(sigma, 4))
            print("σ_err  =", np.round(sigma_err, 6))
            print()

        elif name == 'LHC Filipovic':
            idx = 0
            kappa = final_param[idx:idx + m]; idx += m
            theta = final_param[idx:idx + m]; idx += m
            gamma1 = final_param[idx]; idx += 1

            print("κ      =", np.round(kappa, 4))
            print("θ      =", np.round(theta, 4))
            print("γ₁     =", np.round(gamma1, 4))
            print()

        elif name == 'CIR':
            idx = 0
            kappa = final_param[idx:idx + m]; idx += m
            theta = final_param[idx:idx + m]; idx += m
            sigma = final_param[idx:idx + m]; idx += m
            lambda_i = final_param[idx:idx + m]; idx += m
            sigma_err = final_param[idx] if idx < len(final_param) else np.nan

            print("κ      =", np.round(kappa, 4))
            print("θ      =", np.round(theta, 4))
            print("σ      =", np.round(sigma, 4))
            print("lambda     =", np.round(lambda_i, 4))
            print("σ_err  =", np.round(sigma_err, 6))
            print()




    X_dims = [1,2,3]
    X_dims =  [1,2,3]
    for firm in firms:

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

        directory = f"./Results/{firm}"
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
                ax.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.15),  
                    ncol=3,                       
                    fontsize=12,                  
                    frameon=True
                )       
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
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),  
                ncol=3,                       
                fontsize=12,                  
                frameon=True
            )            
            ax.grid()
            fig.tight_layout()
            fig.savefig(os.path.join(directory, f"DefaultIntensities_Xdim{i}.png"), dpi=150)
            plt.close(fig)


            # Recreated Spreads
            maturities = [1,2, 3,4, 5, 7, 10]  # or ['1Y','3Y','5Y','7Y','10Y']

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
                ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),  
                ncol=2,                       
                fontsize=12,                  
                frameon=True
                )       

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
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),  
                ncol=3,                       
                fontsize=12,                  
                frameon=True
            )       
            fig.tight_layout()
            fig.savefig(os.path.join(directory, f"SurvivalProb_Xdim{i}.png"), dpi=150)
            plt.close(fig)



            ### Compute Global measures of fit.
            # Stack together CDS frames

            models = [CDS_LHC,ZnLHCK,CDS_cirCIR] # stacked fitted CDS spreads.
            # models = [CDS_LHC,CDS_cirCIR] # stacked fitted CDS spreads.

            gfm = global_fit_measures(CDS_obs, models)

            rmse_series, rmse = gfm.rmse()
            ape_series, ape = gfm.ape()
            aae_series, aae = gfm.aae()
            arpe_series, arpe = gfm.arpe()

            # Example structure:
            cols_names = [f"LHCC({i}) Filipovic",f"LHCC({i}) Kalman",f"AFC({i}) Kalman"]
            # cols_names = [f"LHC Filipovic,m={i}",f"CIR Kalman,m={i}"]

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
            handles, labels = axes[0].get_legend_handles_labels()

            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=3,
                fontsize=12,
                frameon=True
            ) 
            fig.suptitle("Risk Measure Time Series by Model", fontsize=14, fontweight="bold")
            # fig.tight_layout(rect=[0, 0, 1, 0.96])
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
            full_global_summary[firm][f'X_dim{i}'] = global_summary



    ### FINALLY, CREATE TABLE FOR GLOBAL RMSE. ON GLOBAL SUMMARY
    metrics = ['RMSE', 'APE', 'AAE', 'ARPE']

    def generate_latex_table(data, metrics):
        latex_str = "\\begin{table}[h!]\n"
        latex_str += "\\centering\n"
        # Column format: Firm | Method | RMSE | APE | AAE | ARPE
        latex_str += "\\begin{tabular}{ll|cccc}\n"
        latex_str += "\\toprule\n"
        
        # Header Row
        latex_str += "& & \\multicolumn{4}{c}{Error Metrics} \\\\\n"
        latex_str += "\\cmidrule(lr){3-6}\n"
        latex_str += "Firm & Method & " + " & ".join(metrics) + " \\\\\n"
        latex_str += "\\midrule\n"

        first_firm = True
        for firm, dimensions in data.items():
            if not first_firm:
                latex_str += "\\midrule\n" # Separator between firms

           
            for dim_name, results in dimensions.items():
                first_row_in_dim = True
                
                for i in range(results.shape[0]):
                    row = []
                    # Only put the firm name on the first row of the firm's data
                    if first_row_in_dim:
                        # Use \multirow to center the firm name across the rows
                        num_rows = len(results) 
                        row.append(f"\\multirow{{{num_rows}}}{{*}}{{{firm}}}")
                        first_row_in_dim = False
                    else:
                        row.append("") # Empty cell for subsequent rows

                    row.append(results.index[i])
                    # Format numerical values to 4 decimal places
                    row.extend([f"{v:.4f}" for v in results.loc[results.index[i]].values])
                    
                    latex_str += " & ".join(row) + " \\\\\n"
                latex_str += "\\midrule\n" # Separator between firms
  
            first_firm = False

        latex_str += "\\bottomrule\n"
        latex_str += "\\end{tabular}\n"
        latex_str += "\\caption{Summary of Fit Metrics by Firm and Model}\n"
        latex_str += "\\label{tab:fit_summary}\n"
        latex_str += "\\end{table}\n"
        
        return latex_str

    # # Split dictionary:
    # primary = {'DANBNK':full_global_summary['DANBNK'],'MONTE':full_global_summary['MONTE']}
    # secondary =  {'CMZB':full_global_summary['DANBNK'],'SVSKHB':full_global_summary['SVSKHB']}
    # # Print main table
    # print(generate_latex_table(primary, metrics))


    # # Print table for appendix
    # print(generate_latex_table(secondary, metrics))


    # # Output Latex Tables:

    # # Redo the plots, but send only relevant to Chpt7 folder


    #### Preliminary investigation.
    sub_df = pd.read_excel("./Data/subset_data.xlsx")
    firms = ['DANBNK','MONTE'] # IG,IG,HY,IG
    # firms = ['SVSKHB'] # IG,IG,HY,IG



    # X_dims = [1,2,3]
    X_dims = [1,2,3]

    for firm in firms:

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


        for i in X_dims:
            directory = f"./Results/{firm}"

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

            ## Print statement controlling solutions are in correct range.
            lhc = LHC_single(0.00248,0.4,0.25)
            lhc.initialise_LHC(1,i,0.5)
            lhc.flatten_params()
            lhc.unflatten_params(final_paramLHC)
            print(f'Filipovic LHCC constr. satisfied: {np.all(lhc.build_constraints(final_paramLHC)>0)}')
            # Remember, in constr. model gamma should not be used as input.
            final_paramLHCK_test = np.append(final_paramLHCK[:2*i],final_paramLHCK[2*i+1:])
            print(f'Kalman LHCC constr. satisfied: {np.all(nonlinear_constraints(final_paramLHCK_test,i)>0)}')
            cir = CIRIntensity(0.0024,0.4,0.25,X_dim=i,cascading=True)
            cir.set_params(final_paramCIR)
            print(f'AFC constr satisfied {np.all(cir.feller_constraint(final_paramCIR)>0)}')
            # Plot dir.
            directory = f"./Results/Chapter7"


            # Loop through maturities and make a separate plot for each
            for j in range(XnLHC.T.shape[1]):
                fig, ax = plt.subplots(figsize=(10, 6))

                ax.plot(test_df['Date'], XnLHCK[:,j], "-", alpha=0.7, color='blue', label=f"LHC Kalman")
                ax.plot(test_df['Date'], XnLHC.T[:,j], "-", alpha=0.7, color='red', label=f"LHC Filipovic")
                ax.plot(test_df['Date'], XnCIR[:,j], "-", alpha=0.7, color='green', label=f"CIR Kalman")

                ax.grid(True)
                ax.set_xlabel("Time (years)")
                ax.set_ylabel("Model States")
                # ax.set_title(f"Model States X{j+1} maturity ({firm})")
                ax.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.15),  
                    ncol=3,                       
                    fontsize=12,                  
                    frameon=True
                )       
                fig.tight_layout()
                save_file = os.path.join(directory, f"{firm}_States_Xdim{i}_X{j+1}.png")
                fig.savefig(save_file, dpi=150)
                plt.close(fig)


            # Default intensities
            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(test_df['Date'], Default_intensityLHCK , "-", alpha=0.7, color='blue', label=f"LHC Kalman")
            ax.plot(test_df['Date'], Default_intensityLHC , "-", alpha=0.7, color='red', label=f"LHC Filipovic")
            ax.plot(test_df['Date'], default_intensityCIR , "-", alpha=0.7, color='green', label=f"CIR Kalman")



            ax.set_xlabel("Time (years)")
            ax.set_ylabel("Default Intensity")
            # ax.set_title(f"Default intensities in different models, {firm}")
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),  
                ncol=3,                       
                fontsize=12,                  
                frameon=True
            )       
            ax.grid()
            fig.tight_layout()
            fig.savefig(os.path.join(directory, f"{firm}_DefaultIntensities_Xdim{i}.png"), dpi=150)
            plt.close(fig)


            # Recreated Spreads
            maturities = [1,2, 3,4, 5, 7, 10]  # or ['1Y','3Y','5Y','7Y','10Y']
            fig, ax = plt.subplots(figsize=(10,6))

            # Loop through maturities and make a separate plot for each
            ax.plot(test_df['Date'], ZnLHCK[:,4], "-", alpha=0.7, color='blue', label="LHC Kalman")
            ax.plot(test_df['Date'], CDS_LHC[:,4], "-", alpha=0.7, color='red', label="LHC Filipovic")
            ax.plot(test_df['Date'], CDS_cirCIR[:,4], "-", alpha=0.7, color='green', label="CIR Kalman")
            ax.plot(test_df['Date'], CDS_obs[:,4], "o", alpha=0.5, color='black', label="Observations")

            ax.grid(True)
            ax.set_xlabel("Time (years)")
            ax.set_ylabel("5Y Spreads")
            # ax.set_title(f"Model Spreads at {5}-year maturity {firm}")
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),  
                ncol=3,                       
                fontsize=12,                  
                frameon=True
            )       
            fig.tight_layout()
            save_file = os.path.join(directory, f"{firm}_Spreads_Xdim{i}_{5}Y.png")
            fig.savefig(save_file, dpi=150)
            plt.close(fig)


            # Survival process.
            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(test_df['Date'], YnLHCK , "-", alpha=0.7, color='blue', label=f"LHC Kalman")
            ax.plot(test_df['Date'], YnLHC, "-", alpha=0.7, color='red', label=f"LHC Filipovic")
            ax.plot(test_df['Date'], YnCIR , "-", alpha=0.7, color='green', label=f"CIR Kalman")


            ax.grid()
            ax.set_xlabel("Time (years)")
            ax.set_ylabel("Survival Probabilities")
            # ax.set_title(f"Survival probabilities in different models, {firm}")
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),  
                ncol=3,                       
                fontsize=12,                  
                frameon=True
            )       
            fig.tight_layout()
            fig.savefig(os.path.join(directory, f"{firm}_SurvivalProb_Xdim{i}.png"), dpi=150)
            plt.close(fig)



            ### Compute Global measures of fit.
            # Stack together CDS frames

            models = [CDS_LHC,ZnLHCK,CDS_cirCIR] # stacked fitted CDS spreads.
            # models = [CDS_LHC,CDS_cirCIR] # stacked fitted CDS spreads.

            gfm = global_fit_measures(CDS_obs, models)

            rmse_series, rmse = gfm.rmse()
            ape_series, ape = gfm.ape()
            aae_series, aae = gfm.aae()
            arpe_series, arpe = gfm.arpe()

            # Example structure:
            cols_names = [f"LHCC({i}) Filipovic",f"LHCC({i}) Kalman",f"AFC({i}) Kalman"]
            # cols_names = [f"LHC Filipovic,m={i}",f"CIR Kalman,m={i}"]
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
            colors = {cols_names[0]:'red',cols_names[1]:'blue',cols_names[2]:'green'}
            # colors = {cols_names[0]:'red',cols_names[1]:'green'}

            for ax, (label, df) in zip(axes, metrics):
                for col in df.columns:
                    ax.plot(test_df['Date'], df[col], label=col, lw=1.5,color=colors[col])
                ax.set_title(label, fontsize=12, fontweight="bold")
                ax.set_xlabel("Observation")
                ax.set_ylabel(label)
                ax.grid(True, alpha=0.3)

            # Add legend (outside for clarity)
            handles, labels = axes[0].get_legend_handles_labels()

            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=3,
                fontsize=12,
                frameon=True
               )       
            # fig.suptitle("Risk Measure Time Series by Model", fontsize=14, fontweight="bold")
            # fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(os.path.join(directory, f"{firm}_Global_fit_errors_Xdim{i}.png"), dpi=150)
            plt.close(fig)

            # === 2) Global summary table ===
            # If rmse, ape, aae, arpe are 1D arrays/lists
            global_summary = pd.DataFrame({
                "RMSE": np.ravel(rmse),
                "APE": np.ravel(ape),
                "AAE": np.ravel(aae),
                "ARPE": np.ravel(arpe)
            }, index=cols_names)
            full_global_summary[firm][f'X_dim{i}'] = global_summary
        # Split dictionary:
    primary = {'DANBNK':full_global_summary['DANBNK'],'MONTE':full_global_summary['MONTE']}
    secondary =  {'CMZB':full_global_summary['DANBNK'],'SVSKHB':full_global_summary['SVSKHB']}
    # Print main table
    metrics = ['RMSE', 'APE', 'AAE', 'ARPE']

    print(generate_latex_table(primary, metrics))


    # Print table for appendix
    print(generate_latex_table(secondary, metrics))


    #### MOVE FINAL OPTION PLOTS TO HERE.
    firms = ['DANBNK','MONTE'] # IG,IG,HY,IG

    X_dims = [1,2,3] # [1,2,3]
    # Run 1-2 first. 3 might be very slow. 
    # X_dims = [1,2,3]
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

        directory = f"./Results/{firm}"
        for i in X_dims:
            filepath = os.path.join(directory, f"Kalman_resultsLHC_NX{i}_gamma1.npz")
            data = np.load(filepath)
            final_paramLHCK = data["final_param"]
            XnLHCK = data["Xn"]
            YnLHCK=data["Yn"]
            PnLHC = data["Pn"]
            ZnLHCK = data["CDS_model"]
            Default_intensityLHCK = data["Default_intensity"]
            print_model_params("LHC Kalman", final_paramLHCK, m=i)


            ######## Option Pricing in the Models. Here, we can practically only use Kalman-like models
            r,delta,tenor = 0.00248,0.4, 0.25
            # Initalise LHC model for pricing.
            ## The inital value at en of data is just observed CDS spread.
            cds0 = CDS_obs[-1,2]

            # Then strikes around the above. Say PM 150 Bps.
            # strike_diff_grid = np.linspace(-60,60,20) / 10000
            strike_diff_grid = np.linspace(-40,120,int((120+40)/20)+1) / 10000
            strikes = (cds0 + strike_diff_grid).flatten()
            # Set Simulate option sizes.
            chi0 = np.concatenate((np.array([1]), XnLHCK[-1,:]))
            # Set discretization and number of simul. 1m simuls, fairly coarse grid
            N, M = 20000, 100
            # N, M = 400,100
            seed = 2000
            # Take Legendre poly at 20. 
            leg_deg = 20 # 20 . This is were it explodes in X_dim=3
       
        
            # Read in previous data.
            option_filepath = os.path.join(directory, f"Option_data_{firm}_X{i}.npz")

            # Ensure directory exists
            option_data = np.load(option_filepath)
            cdso_actual=option_data['cdso_poly']
            cdso_MC=option_data['cdso_lhc_MC']
            cdso_MC_cir=option_data['cdso_cir_MC']
            digital_MC=option_data['digital_lhc']
            digital_MC_cir=option_data['digital_cir_MC']              
            look_MC=option_data['look_MC_lhc']
            look_MC_cir=option_data['look_MC_cir']   
            look_cds_min=option_data['look_lhc_min']            
            look_cds_min_cir=option_data['look_cir_min']            
            strikes=option_data['cdso_strikes']
            barriers=option_data['digital_barriers']
            cds0=option_data['cds_obs'] 
            
            strike_offsets_bps = (strike_diff_grid * 1e4).flatten()     # offsets in bps
            strikes_bps = (strikes * 1e4).flatten()                     # total strikes in bps
            cdso_MC_bps = cdso_MC * 1e4
            # cdso_MC_hist_bps = cdso_MC_hist * 1e4
            cdso_MC_bps_cir = cdso_MC_cir * 1e4
            # cdso_MC_hist_bps_cir = cdso_MC_hist_cir * 1e4
            # cdso_MC_bps_cir_fourier = cdso_fourier * 1e4
            option_filepath = os.path.join(directory, f"Option_data_{firm}_X{i}_full.npz")

            # Ensure directory exists
            option_data_full = np.load(option_filepath)
            ## get the full data:
            cdso_MC_g = option_data_full['cdso_lhc_MC']
            digital_MC_g = option_data_full['digital_lhc']
            look_MC_g = option_data_full['look_MC_lhc']
            look_cds_min_g = option_data_full['look_lhc_min'] 


            save_path = f"./Results/Options_full"

        #     save_path = directory + f"/Options"
            
            os.makedirs(save_path, exist_ok=True)

            # --- Plot 1: Price vs Strike Offset ---
            fig1, ax1 = plt.subplots(figsize=(8, 6))

            ax1.plot(strike_offsets_bps, cdso_MC_bps, 'o-', color='navy', alpha=0.8,label='LHC Model, Simulated')
            ax1.plot(strike_offsets_bps, cdso_actual*10000, 'o-', color='red', alpha=0.8,label=f'LHC Model, Legendre n={leg_deg}')
            ax1.plot(strike_offsets_bps, cdso_MC_g*10000, 'x-', color='magenta', alpha=0.8,label='LHC Model full, Simulated')
            # ax1.plot(strike_offsets_bps, cdso_actual_g*10000, 'o-', color='magenta', alpha=0.8,label=f'LHC Model full, Legendre n={leg_deg}')
            ax1.plot(strike_offsets_bps, cdso_MC_bps_cir, 'o-', color='forestgreen', alpha=0.8,label='CIR Model, Simulated')
            # ax1.plot(strike_offsets_bps, cdso_MC_bps_cir_fourier, 'o-', color='grey', alpha=0.8,label='CIR Model, Fourier')

            ax1.set_xlabel("Strike Offset (bps)")
            ax1.set_ylabel("CDS Option Price (bps)")
            # ax1.set_title("CDS Option Price vs Strike Offset")
            ax1.grid(True)
            ax1.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),  
                ncol=2,                       
                fontsize=12,                  
                frameon=True
            )       
            fig1.tight_layout()
            fig1.savefig(os.path.join(save_path, f"CDSO_MC_vs_Strike_{firm}_m{i}.png"), dpi=150)
            plt.close(fig1)


            # --- Plot 1: Price vs Strike Offset ---
            # get barrier again
            barriers_percentage = barriers / cds0
            fig1, ax1 = plt.subplots(figsize=(8, 6))

            ax1.plot(barriers_percentage, digital_MC, 'o-', color='navy', alpha=0.8,label='LHC Model, Simulated')
            ax1.plot(barriers_percentage, digital_MC_g, 'o-', color='magenta', alpha=0.8,label='LHC Model full, Simulated')
            ax1.plot(barriers_percentage, digital_MC_cir, 'o-', color='forestgreen', alpha=0.8,label='CIR Model, Simulated')
            # ax1.plot(strike_offsets_bps, cdso_MC_bps_cir_fourier, 'o-', color='grey', alpha=0.8,label='CIR Model, Fourier')

            ax1.set_xlabel("Percentage of spot CDS rate")
            ax1.set_ylabel("Digital barrier Option Price (bps)")
            # ax1.set_title("Digital barrier option Price")
            ax1.grid(True)
            ax1.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),  
                ncol=2,                       
                fontsize=12,                  
                frameon=True
            )       
            fig1.tight_layout()
            fig1.savefig(os.path.join(save_path, f"Digital_MC_vs_Strike_{firm}_m{i}.png"), dpi=150)
            plt.close(fig1)


            print(f'Lookback LHC: {look_MC*10000}')
            print(f'Lookback LHC full: {look_MC_g*10000}')
            print(f'Lookback CIR: {look_MC_cir*10000}')

            # Store this in a 


            print(f'Lookback LHC: {look_MC*10000}')
            print(f'Lookback LHC full: {look_MC_g*10000}')
            print(f'Lookback CIR: {look_MC_cir*10000}')

            fig2, ax2 = plt.subplots(figsize=(8, 6))

            vals = {
                "LHCC":        (look_MC * 10000, "orange", 0.02),
                "LHCC gamma":  (look_MC_g * 10000, "magenta", 0.20),  # shifted right
                "CIR":         (look_MC_cir * 10000, "green", 0.40),
            }

            for label, (y, color, x) in vals.items():
                ax2.text(
                    x, y,
                    f"{y:.2f} bps",
                    color=color,
                    fontsize=9,
                    va='bottom',
                    ha='left',
                    fontweight='bold',
                    zorder=10
                )

            ax2.set_ylim(0, max(v[0] for v in vals.values()) * 1.1)
            # ax2.set_title("Lookback prices")
            ax2.grid(True)
            # ax2.set_title("Lookback prices")
            ax2.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),  
                ncol=3,                       
                fontsize=12,                  
                frameon=True
            )       
            fig2.tight_layout()
            fig2.savefig(os.path.join(save_path, f"Lookback_MC_convergence_{firm}_m{i}.png"), dpi=150)
            plt.close(fig2)


            # Minimum distribution.
            fig4, ax4 = plt.subplots(figsize=(8, 6))

            # Plot histograms (normalized to probability densities)
            ax4.hist(look_cds_min_cir * 10000, bins=40, alpha=0.6, density=True,
                    label='AFC Model', color='forestgreen', edgecolor='black')
            ax4.hist(look_cds_min * 10000, bins=40, alpha=0.6, density=True,
                    label='LHCC Model', color='navy', edgecolor='black')
            ax4.hist(look_cds_min_g* 10000, bins=40, alpha=0.6, density=True,
                    label='LHCC Model,full', color='magenta', edgecolor='black')

            # Axis labels and title
            ax4.set_xlabel("Minimum CDS (bps)")
            ax4.set_ylabel("Density")
            # ax4.set_title("Distribution of Minimum CDS spreads")

            # Add legend and grid
            ax4.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),  
                ncol=3,                       
                fontsize=12,                  
                frameon=True
            )       
            ax4.grid(True, linestyle='--', alpha=0.6)

            # Tight layout and save
            fig4.tight_layout()
            fig4.savefig(os.path.join(save_path, f"Lookback_Distribution_Comparison_{firm}_m{i}.png"), dpi=150)
            plt.close(fig4)










#### Add a volatility matrix plot to compare the diffusion functions and drift functions of the
#### default
    sub_df = pd.read_excel("./Data/subset_data.xlsx")
    firms = ['DANBNK','MONTE'] # IG,IG,HY,IG
    # firms = ['SVSKHB'] # IG,IG,HY,IG

    X_dims = [1,2,3]

    for firm in firms:

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


        for i in X_dims:
            directory = f"./Results/{firm}"

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

            # Plot dir.
            directory = f"./Results/Chapter7"

            ## Get sigma_AFC:
            sigma1_afc = final_paramCIR[2*i]

            sigma1_lhc = final_paramLHCK[3*i+1]
            gamma1 = final_paramLHCK[2*i]
            # Get default intensity process of 

            lambda_t = Default_intensityLHCK

            # plot
            plt.plot(t,sigma1_lhc * np.sqrt((gamma1-Default_intensityLHCK)*Default_intensityLHCK),color='black',label='LHCC')
            plt.plot(t,  sigma1_afc*np.sqrt(default_intensityCIR), label='CIR')
            plt.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),  
                ncol=3,                       
                fontsize=12,                  
                frameon=True
            )       
            plt.show()





######################### APPENDIX Add analysis for w gamma1 run. 
  #### Preliminary investigation.
    sub_df = pd.read_excel("./Data/subset_data.xlsx")
    firms = ['DANBNK','MONTE'] # IG,IG,HY,IG
    # firms = ['SVSKHB'] # IG,IG,HY,IG



    # X_dims = [1,2,3]
    X_dims = [1,2,3]

    for firm in firms:

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


        for i in X_dims:
            directory = f"./Results/{firm}"

            filepath = os.path.join(directory, f"Kalman_resultsLHC_NX{i}.npz")
            data = np.load(filepath)
            final_paramLHCK = data["final_param"]
            XnLHCK = data["Xn"]
            YnLHCK=data["Yn"]
            PnLHC = data["Pn"]
            ZnLHCK = data["CDS_model"]
            Default_intensityLHCK = data["Default_intensity"]
            print_model_params("LHC Kalman", final_paramLHCK, m=i)

            filepath = os.path.join(directory, f"Kalman_resultsLHC_NX{i}_gamma1.npz")
            data = np.load(filepath)
            final_paramLHCK_g = data["final_param"]
            XnLHCK_g = data["Xn"]
            YnLHCK_g=data["Yn"]
            PnLHC_g = data["Pn"]
            ZnLHCK_g = data["CDS_model"]
            Default_intensityLHCK_g = data["Default_intensity"]
            print_model_params("LHC Kalman", final_paramLHCK_g, m=i)


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

            ## Print statement controlling solutions are in correct range.
            # Remember, in constr. model gamma should not be used as input.
            print(f'Kalman LHCC gamma1 constr. satisfied: {np.all(nonlinear_constraints_wgamma(final_paramLHCK_g,i)>0)}')
            

            # Plot dir.
            directory = f"./Results/wGamma1"


            # Loop through maturities and make a separate plot for each
            for j in range(XnLHC.T.shape[1]):
                fig, ax = plt.subplots(figsize=(10, 6))

                ax.plot(test_df['Date'], XnLHCK[:,j], "-", alpha=0.7, color='blue', label=f"LHC Kalman, restricted")
                ax.plot(test_df['Date'], XnLHCK_g[:,j], "-", alpha=0.7, color='magenta', label=f"LHC Kalman, unrestricted")
                ax.plot(test_df['Date'], XnLHC.T[:,j], "-", alpha=0.7, color='red', label=f"LHC Filipovic")
                ax.plot(test_df['Date'], XnCIR[:,j], "-", alpha=0.7, color='green', label=f"CIR Kalman")

                ax.grid(True)
                ax.set_xlabel("Time (years)")
                ax.set_ylabel("Model States")
                # ax.set_title(f"Model States X{j+1} maturity ({firm})")
                ax.legend()

                fig.tight_layout()
                save_file = os.path.join(directory, f"{firm}_States_Xdim{i}_X{j+1}.png")
                fig.savefig(save_file, dpi=150)
                plt.close(fig)


            # Default intensities
            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(test_df['Date'], Default_intensityLHCK , "-", alpha=0.7, color='blue', label=f"LHC Kalman, restricted")
            ax.plot(test_df['Date'], Default_intensityLHCK_g , "-", alpha=0.7, color='magenta', label=f"LHC Kalman, unrestricted")            
            ax.plot(test_df['Date'], Default_intensityLHC , "-", alpha=0.7, color='red', label=f"LHC Filipovic")
            ax.plot(test_df['Date'], default_intensityCIR , "-", alpha=0.7, color='green', label=f"CIR Kalman")



            ax.set_xlabel("Time (years)")
            ax.set_ylabel("Default Intensity")
            # ax.set_title(f"Default intensities in different models, {firm}")
            ax.legend()
            ax.grid()
            fig.tight_layout()
            fig.savefig(os.path.join(directory, f"{firm}_DefaultIntensities_Xdim{i}.png"), dpi=150)
            plt.close(fig)


            # Recreated Spreads
            maturities = [1,2, 3,4, 5, 7, 10]  # or ['1Y','3Y','5Y','7Y','10Y']
            fig, ax = plt.subplots(figsize=(10,6))

            # Loop through maturities and make a separate plot for each
            ax.plot(test_df['Date'], ZnLHCK[:,4], "-", alpha=0.7, color='blue', label="LHC Kalman, restricted")
            ax.plot(test_df['Date'], ZnLHCK_g[:,4], "-", alpha=0.7, color='magenta', label="LHC Kalman, unrestricted")
            ax.plot(test_df['Date'], CDS_LHC[:,4], "-", alpha=0.7, color='red', label="LHC Filipovic")
            ax.plot(test_df['Date'], CDS_cirCIR[:,4], "-", alpha=0.7, color='green', label="CIR Kalman")
            ax.plot(test_df['Date'], CDS_obs[:,4], "o", alpha=0.5, color='black', label="Observations")

            ax.grid(True)
            ax.set_xlabel("Time (years)")
            ax.set_ylabel("5Y Spreads")
            # ax.set_title(f"Model Spreads at {5}-year maturity {firm}")
            ax.legend()

            fig.tight_layout()
            save_file = os.path.join(directory, f"{firm}_Spreads_Xdim{i}_{5}Y.png")
            fig.savefig(save_file, dpi=150)
            plt.close(fig)


            # Survival process.
            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(test_df['Date'], YnLHCK , "-", alpha=0.7, color='blue', label=f"LHC Kalman, restricted")
            ax.plot(test_df['Date'], YnLHCK_g , "-", alpha=0.7, color='magenta', label=f"LHC Kalman, unrestricted")           
            ax.plot(test_df['Date'], YnLHC, "-", alpha=0.7, color='red', label=f"LHC Filipovic")
            ax.plot(test_df['Date'], YnCIR , "-", alpha=0.7, color='green', label=f"CIR Kalman")


            ax.grid()
            ax.set_xlabel("Time (years)")
            ax.set_ylabel("Survival Probabilities")
            # ax.set_title(f"Survival probabilities in different models, {firm}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(directory, f"{firm}_SurvivalProb_Xdim{i}.png"), dpi=150)
            plt.close(fig)



            ### Compute Global measures of fit.
            # Stack together CDS frames

            models = [CDS_LHC,ZnLHCK,ZnLHCK_g,CDS_cirCIR] # stacked fitted CDS spreads.
            # models = [CDS_LHC,CDS_cirCIR] # stacked fitted CDS spreads.

            gfm = global_fit_measures(CDS_obs, models)

            rmse_series, rmse = gfm.rmse()
            ape_series, ape = gfm.ape()
            aae_series, aae = gfm.aae()
            arpe_series, arpe = gfm.arpe()

            # Example structure:
            cols_names = [f"LHCC({i}) Filipovic",f"LHCC({i}) Kalman, restricted",
                          f"LHCC({i}) Kalman, unrestricted",f"AFC({i}) Kalman"]
            # cols_names = [f"LHC Filipovic,m={i}",f"CIR Kalman,m={i}"]

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
            colors = {cols_names[0]:'red',cols_names[1]:'blue',
                      cols_names[2]:'magenta',cols_names[3]:'green'}
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
            
            # fig.suptitle("Risk Measure Time Series by Model", fontsize=14, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(os.path.join(directory, f"{firm}_Global_fit_errors_Xdim{i}.png"), dpi=150)
            plt.close(fig)

            # === 2) Global summary table ===
            # If rmse, ape, aae, arpe are 1D arrays/lists
            global_summary = pd.DataFrame({
                "RMSE": np.ravel(rmse),
                "APE": np.ravel(ape),
                "AAE": np.ravel(aae),
                "ARPE": np.ravel(arpe)
            }, index=cols_names)
            full_global_summary[firm][f'X_dim{i}'] = global_summary
        # Split dictionary:
    primary = {'DANBNK':full_global_summary['DANBNK'],'MONTE':full_global_summary['MONTE']}
    secondary =  {'CMZB':full_global_summary['DANBNK'],'SVSKHB':full_global_summary['SVSKHB']}
    # Print main table
    metrics = ['RMSE', 'APE', 'AAE', 'ARPE']

    print(generate_latex_table(primary, metrics))


    # Print table for appendix
    print(generate_latex_table(secondary, metrics))



    stopper = 1