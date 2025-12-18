
#### Plots for comparison. TODO: Move to other file.
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from Models.LHCModels.LHC_single import LHC_single,nonlinear_constraints_mpr, rebuild_lhc_struct,kalmanfilter_opt,build_P_params
from Models.LHCModels.LHC_wGamma1 import LHC_single as LHC_wGamma1
from Models.LHCModels.LHC_wGamma1 import nonlinear_constraints_mpr as nonlinear_constraints_mpr_wgamma

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
    from Result_plots import print_model_params

    X_dims = [1,2,3]

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
            directory = f"./Results_MPR/{firm}"

            filepath = os.path.join(directory, f"Kalman_resultsLHC_NX{i}.npz")
            data = np.load(filepath)
            final_paramLHCK = data["final_param"]
            XnLHCK = data["Xn"]
            YnLHCK=data["Yn"]
            PnLHC = data["Pn"]
            ZnLHCK = data["CDS_model"]
            Default_intensityLHCK = data["Default_intensity"]
            print_model_params("LHC Kalman", final_paramLHCK, m=i)

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


            cir = CIRIntensity(0.0024,0.4,0.25,X_dim=i,cascading=True)
            cir.set_params(final_paramCIR)
            print(f'AFC constr satisfied {np.all(cir.feller_constraint_mpr(final_paramCIR)>0)}')
            # Plot dir.
            directory = f"./Results_MPR/Chapter7"


            # Loop through maturities and make a separate plot for each
            for j in range(XnLHCK.shape[1]):
                fig, ax = plt.subplots(figsize=(10, 6))

                ax.plot(test_df['Date'], XnLHCK[:,j], "-", alpha=0.7, color='blue', label=f"LHC Kalman")
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
            ax.plot(test_df['Date'], Default_intensityLHCK , "-", alpha=0.7, color='blue', label=f"LHC Kalman")
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

            # Example list of maturities (adjust names if your columns differ)
            maturities = [1,2, 3,4, 5, 7, 10]  # or ['1Y','3Y','5Y','7Y','10Y']
            fig, ax = plt.subplots(figsize=(10,6))

            # Loop through maturities and make a separate plot for each
            ax.plot(test_df['Date'], ZnLHCK[:,4], "-", alpha=0.7, color='blue', label="LHC Kalman")
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
            ax.plot(test_df['Date'], YnLHCK , "-", alpha=0.7, color='blue', label=f"LHC Kalman")
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

            models = [ZnLHCK,CDS_cirCIR] # stacked fitted CDS spreads.
            # models = [CDS_LHC,CDS_cirCIR] # stacked fitted CDS spreads.

            gfm = global_fit_measures(CDS_obs, models)

            rmse_series, rmse = gfm.rmse()
            ape_series, ape = gfm.ape()
            aae_series, aae = gfm.aae()
            arpe_series, arpe = gfm.arpe()

            # Example structure:
            cols_names = [f"LHCC({i}) Kalman",f"AFC({i}) Kalman"]
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
            colors = {cols_names[0]:'blue',cols_names[1]:'green'}
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
            directory = f"./Results_MPR/{firm}"

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
            print(f'Kalman LHCC gamma1 constr. satisfied: {np.all(nonlinear_constraints_mpr_wgamma(final_paramLHCK_g,i)>0)}')
            

            # Plot dir.
            directory = f"./Results_MPR/wGamma1"


            # Loop through maturities and make a separate plot for each
            for j in range(XnLHCK.shape[1]):
                fig, ax = plt.subplots(figsize=(10, 6))

                ax.plot(test_df['Date'], XnLHCK[:,j], "-", alpha=0.7, color='blue', label=f"LHC Kalman, restricted")
                ax.plot(test_df['Date'], XnLHCK_g[:,j], "-", alpha=0.7, color='magenta', label=f"LHC Kalman, unrestricted")
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

            # Example list of maturities (adjust names if your columns differ)
            maturities = [1,2, 3,4, 5, 7, 10]  # or ['1Y','3Y','5Y','7Y','10Y']
            fig, ax = plt.subplots(figsize=(10,6))

            # Loop through maturities and make a separate plot for each
            ax.plot(test_df['Date'], ZnLHCK[:,4], "-", alpha=0.7, color='blue', label="LHC Kalman, restricted")
            ax.plot(test_df['Date'], ZnLHCK_g[:,4], "-", alpha=0.7, color='magenta', label="LHC Kalman, unrestricted")
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

            models = [ZnLHCK,ZnLHCK_g,CDS_cirCIR] # stacked fitted CDS spreads.
            # models = [CDS_LHC,CDS_cirCIR] # stacked fitted CDS spreads.

            gfm = global_fit_measures(CDS_obs, models)

            rmse_series, rmse = gfm.rmse()
            ape_series, ape = gfm.ape()
            aae_series, aae = gfm.aae()
            arpe_series, arpe = gfm.arpe()

            # Example structure:
            cols_names = [f"LHCC({i}) Kalman, restricted",
                          f"LHCC({i}) Kalman, unrestricted",f"AFC({i}) Kalman"]
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
            colors = {cols_names[0]:'blue',
                      cols_names[1]:'magenta',cols_names[2]:'green'}
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

