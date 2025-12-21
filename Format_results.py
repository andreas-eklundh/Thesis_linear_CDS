import pandas as pd
import os
import numpy as np
from Models.LHCModels.LHC_wGamma1 import LHC_single as LHC_wGamma1
from Models.LHCModels.LHC_single import LHC_single as LHC_single
from Models.BaselineCIR_alternative.CIR_Multifactor import CIRIntensity 


# --- Configuration ---
# Map the filenames to the column groups in the table (1, 2, 3)
# Update the filenames if they change. 
# We assume LHCC(3) file might be added later, so it's included in the map but marked optional.
files_map = {
    1: "./Simulation_studies/lhc_parameter_comparison_Xdim1.xlsx",
    2: "./Simulation_studies/lhc_parameter_comparison_Xdim2.xlsx",
    3: "./Simulation_studies/lhc_parameter_comparison_Xdim3.xlsx" # Placeholder name
}


files_map_cir = {
    1: "./Simulation_studies/cir_parameter_comparison_Xdim1.xlsx",
    2: "./Simulation_studies/cir_parameter_comparison_Xdim2.xlsx",
    3: "./Simulation_studies/cir_parameter_comparison_Xdim3.xlsx" # Placeholder name
}
# --- Configuration Part 2 (New) ---
FIRMS = [ 'DANBNK', 'MONTE']
BASE_PATH = r"./Results"

# Define the order of parameters for the Firm tables
# Note: AFC does not have Gamma, handled in logic.
ROW_ORDER_FIRMS = [
    ("kappa1", r"$\kappa_1$"), ("kappa2", r"$\kappa_2$"), ("kappa3", r"$\kappa_3$"),
    ("theta1", r"$\theta_1$"), ("theta2", r"$\theta_2$"), ("theta3", r"$\theta_3$"),
    ("gamma1", r"$\gamma_1$"),
    ("lambda1", r"$\lambda_1$"), ("lambda2", r"$\lambda_2$"), ("lambda3", r"$\lambda_3$"),
    ("sigma1", r"$\sigma_1$"), ("sigma2", r"$\sigma_2$"), ("sigma3", r"$\sigma_3$"),
    ("sigma_err", r"$\sigma_{err}$")
]


# Map LaTeX row labels to the 'Parameter' name found in the CSV files
# Key: LaTeX Label, Value: CSV Parameter Name
row_mapping = {
    r"$\kappa_1$": "kappa1",
    r"$\kappa_2$": "kappa2",
    r"$\kappa_3$": "kappa3",
    r"$\theta_1$": "theta1",
    r"$\theta_2$": "theta2",
    r"$\theta_3$": "theta3",
    r"$\gamma_1$": "gamma1",
    r"$\lambda_1$": "lambda1",
    r"$\lambda_2$": "lambda2",
    r"$\lambda_3$": "lambda3",
    r"$\sigma_1$": "sigma1",
    r"$\sigma_2$": "sigma2",
    r"$\sigma_3$": "sigma3",
    r"$\sigma_{err}$": "sigma_err" # Adjust based on exact string in CSV (e.g., "error", "sigmaerr")
}




# Map LaTeX row labels to the 'Parameter' name found in the CSV files
# Key: LaTeX Label, Value: CSV Parameter Name
row_mapping = {
    r"$\kappa_1$": "kappa1",
    r"$\kappa_2$": "kappa2",
    r"$\kappa_3$": "kappa3",
    r"$\theta_1$": "theta1",
    r"$\theta_2$": "theta2",
    r"$\theta_3$": "theta3",
    r"$\gamma_1$": "gamma1",
    r"$\lambda_1$": "lambda1",
    r"$\lambda_2$": "lambda2",
    r"$\lambda_3$": "lambda3",
    r"$\sigma_1$": "sigma1",
    r"$\sigma_2$": "sigma2",
    r"$\sigma_3$": "sigma3",
    r"$\sigma_{err}$": "sigma_err" # Adjust based on exact string in CSV (e.g., "error", "sigmaerr")
}

# --- Helper Functions ---

def load_data(files_map):
    """Loads CSV files into a dictionary of DataFrames."""
    dfs = {}
    for key, filename in files_map.items():
            # Read CSV
            df = pd.read_excel(filename)
            # Ensure 'Parameter' column is the index for easy lookup
            if 'Parameter' in df.columns:
                df.set_index('Parameter', inplace=True)
            dfs[key] = df
    return dfs

def format_val(val):
    """Format val to 4 decimals, or scientific notation if very small."""
    try:
        if pd.isna(val) or val == "" or val == 0:
            return "-"
        
        v = float(val)

        # Use scientific notation if extremely small
        if abs(v) < 5e-4:
            return f"{v:.2e}"
    
        return f"{v:.3f}"
    
    except:
        return "-"


def get_row_data(param_name_csv, dfs, lhc=True):
    """Extracts True, Kalman, Filipovic values for a specific parameter across all 3 file dimensions."""
    row_cells = []
    
    for i in range(1, 4): # Loop through LHCC(1), LHCC(2), LHCC(3)
        df = dfs.get(i)
        
        # If df exists and parameter is in the index
        if df is not None and param_name_csv in df.index:
            try:
                # Extract values. Column names must match CSV headers exactly
                val_true = format_val(df.loc[param_name_csv, 'True'])
                val_kalman = format_val(df.loc[param_name_csv, 'Estimated Kalman'])
                if lhc:
                    val_filipovic = format_val(df.loc[param_name_csv, 'Filipovic'])
                    val_filipovic = r"\makecell{" +val_filipovic + r"\\[-6pt] \small{}}"

                se = format_val(df.loc[param_name_csv, 'SE'])
                val_kalman = r"\makecell{" +val_kalman + r"\\[-6pt] \small{(" + se + ")}}"
                val_true = r"\makecell{" +val_true + r"\\[-6pt] \small{}}"
                if lhc:
                    row_cells.extend([val_true, val_kalman, val_filipovic])
                else:
                    row_cells.extend([val_true, val_kalman])

            except KeyError as e:
                # Column not found or other error
                if lhc:
                    row_cells.extend(["-", "-", "-"])
                else:
                    row_cells.extend(["-", "-"])
        else:
            if lhc:
                row_cells.extend(["-", "-", "-"])
            else: 
                row_cells.extend([ "-", "-"])
 
    return row_cells

def get_loglike_data(dfs,lhc=True):
    """Extracts LogLikelihood (assumed to be constant per file)."""
    row_cells = []
    for i in range(1, 4):
        df = dfs.get(i)
        if df is not None and 'LogLike' in df.columns:
            # Take the first value as it's usually model-wide
            val = format_val(df['LogLike'].iloc[0])
            if lhc:
                row_cells.extend(["-", val,'-'])
            else:
                row_cells.extend(["-", val])


        else:
            if lhc:
                row_cells.extend(["-", "-", "-"])
            else: 
                row_cells.extend([ "-", "-"])

    return row_cells

# --- Main Execution ---

def generate_latex_lhc():
    dfs = load_data(files_map)
    
    print(r"\begin{table}[H]")
    print(r"    \centering")
    print(r"    \begin{tabular}{|c|ccc|ccc|ccc|}")
    print(r"        \hline")
    print(r"         & \multicolumn{3}{c|}{\textbf{LHCC(1)}} & \multicolumn{3}{c|}{\textbf{LHCC(2)}} & \multicolumn{3}{c|}{\textbf{LHCC(3)}} \\")
    print(r"        \cline{2-10}")
    print(r"         & True & Kalman & Filipovic & True & Kalman & Filipovic & True & Kalman & Filipovic \\")
    print(r"        \hline")

    # Generate rows for parameters
    for latex_label, csv_param in row_mapping.items():
        cells = get_row_data(csv_param, dfs)
        # Join cells with ' & '
        line_content = " & ".join(cells)
        print(f"        {latex_label} & {line_content} \\\\")

    print(r"        \hline")
    
    # Generate Log Likelihood row
    log_cells = get_loglike_data(dfs)
    log_line = " & ".join(log_cells)
    print(f"        $\\log \\mathcal{{L}}$ & {log_line}\\\\")
    
    print(r"        \hline")
    print(r"    \end{tabular}")
    print(r"    \caption{Parameter Estimation in the LHCC Model}")
    print(r"    \label{tab:lhc_simul}")
    print(r"\end{table}")


#### Cascading logic.

def generate_latex_afc(files_map):
    dfs = load_data(files_map_cir)
    
    print(r"\begin{table}[H]")
    print(r"    \centering")
    print(r"        \begin{tabular}{|c|cc|cc|cc|}")
    print(r"        \hline")
    print(r"         & \multicolumn{2}{c|}{\textbf{AFC(1)}} & \multicolumn{2}{c|}{\textbf{AFC(2)}} & \multicolumn{2}{c|}{\textbf{AFC(3)}} \\")
    print(r"        \cline{2-7}")
    print(r"        $\Theta$ & True & Kalman & True & Kalman & True & Kalman \\")
    print(r"        \hline")

    # Generate rows for parameters
    for latex_label, csv_param in row_mapping.items():
        cells = get_row_data(csv_param, dfs,lhc=False)
        # Join cells with ' & '
        line_content = " & ".join(cells)
        print(f"        {latex_label} & {line_content} \\\\")

    print(r"        \hline")
    
    # Generate Log Likelihood row
    log_cells = get_loglike_data(dfs,lhc=False)
    log_line = " & ".join(log_cells)
    print(f"        $\\log \\mathcal{{L}}$ & {log_line}\\\\")
    
    print(r"        \hline")
    print(r"    \end{tabular}")
    print(r"    \caption{Parameter Estimation in the AFC Model}")
    print(r"    \label{tab:cir_simul}")
    print(r"\end{table}")


# --- New Helper Functions for Empirical Data (NPZ) ---

def get_param_value(param_name, param_array, m,model):
    if param_array is None:
        return None


    # Standard params
    p_type = ''.join([i for i in param_name if not i.isdigit()]) # kappa, theta...
    if p_type != 'sigma_err':
        p_idx = int(''.join([i for i in param_name if i.isdigit()])) # 1, 2, 3
        
        # If the specific parameter index (e.g. 3) is greater than current dimension m, it doesn't exist
        if p_idx > m:
            return None

    
    # 0-based index conversion
        idx_in_group = p_idx - 1 

    if model == 'affine':
        if p_type == "kappa":
            flat_idx = 0 + idx_in_group
        elif p_type == "theta":
            flat_idx = m + idx_in_group
        elif p_type == "sigma":
            flat_idx = 2 * m + idx_in_group
        elif p_type == "lambda": 
                flat_idx = 3 * m + idx_in_group
        elif p_type == 'sigma_err':
            return param_array[-1]
        else:
            return None

    elif model == 'lhc': 
        if p_type == "kappa":
            flat_idx = 0 + idx_in_group
        elif p_type == "theta":
            flat_idx = m + idx_in_group
        elif p_type == "gamma":
                # Assuming Gamma is after lambda? Or maybe it's only 1 val?
                # Let's assume it's at the end before sigma_err or specific spot
                flat_idx = 2 * m 
        elif p_type == "lambda": 
                flat_idx = 2 * m + 1+idx_in_group
        elif p_type == "sigma":
            flat_idx = 3 * m + 1+idx_in_group
        elif p_type == 'sigma_err':
            return param_array[-1]
        else:
            return None
    
    elif model == 'lhc_f':
        if p_type == "kappa":
            flat_idx = 0 + idx_in_group
        elif p_type == "theta":
            flat_idx = m + idx_in_group
        elif p_type == "gamma":
                # Assuming Gamma is after lambda? Or maybe it's only 1 val?
                # Let's assume it's at the end before sigma_err or specific spot
                flat_idx = 2 * m 
        else:
            return None
    


    if flat_idx < len(param_array):
        return param_array[flat_idx]
    return None

# --- Main Execution ---

def generate_latex_lhc():
    dfs = load_data(files_map)
    
    print(r"\begin{table}[H]")
    print(r"    \centering")
    print(r"    \begin{tabular}{|c|ccc|ccc|ccc|}")
    print(r"        \hline")
    print(r"         & \multicolumn{3}{c|}{\textbf{LHCC(1)}} & \multicolumn{3}{c|}{\textbf{LHCC(2)}} & \multicolumn{3}{c|}{\textbf{LHCC(3)}} \\")
    print(r"        \cline{2-10}")
    print(r"         & True & Kalman & Filipovic & True & Kalman & Filipovic & True & Kalman & Filipovic \\")
    print(r"        \hline")

    # Generate rows for parameters
    for latex_label, csv_param in row_mapping.items():
        cells = get_row_data(csv_param, dfs)
        # Join cells with ' & '
        line_content = " & ".join(cells)
        print(f"        {latex_label} & {line_content} \\\\")

    print(r"        \hline")
    
    # Generate Log Likelihood row
    log_cells = get_loglike_data(dfs)
    log_line = " & ".join(log_cells)
    print(f"        $\\log \\mathcal{{L}}$ & {log_line}\\\\")
    
    print(r"        \hline")
    print(r"    \end{tabular}")
    print(r"    \caption{Parameter Estimation in the LHCC Model}")
    print(r"    \label{tab:lhc_simul}")
    print(r"\end{table}")


#### Cascading logic.

def generate_latex_afc():
    dfs = load_data(files_map_cir)
    
    print(r"\begin{table}[H]")
    print(r"    \centering")
    print(r"        \begin{tabular}{|c|cc|cc|cc|}")
    print(r"        \hline")
    print(r"         & \multicolumn{2}{c|}{\textbf{AFC(1)}} & \multicolumn{2}{c|}{\textbf{AFC(2)}} & \multicolumn{2}{c|}{\textbf{AFC(3)}} \\")
    print(r"        \cline{2-7}")
    print(r"        & True & Kalman & True & Kalman & True & Kalman \\")
    print(r"        \hline")

    # Generate rows for parameters
    for latex_label, csv_param in row_mapping.items():
        cells = get_row_data(csv_param, dfs,lhc=False)
        # Join cells with ' & '
        line_content = " & ".join(cells)
        print(f"        {latex_label} & {line_content} \\\\")

    print(r"        \hline")
    
    # Generate Log Likelihood row
    log_cells = get_loglike_data(dfs,lhc=False)
    log_line = " & ".join(log_cells)
    print(f"        $\\log \\mathcal{{L}}$ & {log_line}\\\\")
    
    print(r"        \hline")
    print(r"    \end{tabular}")
    print(r"    \caption{Parameter Estimation in the AFC Model}")
    print(r"    \label{tab:cir_simul}")
    print(r"\end{table}")

# --- New Function: Generate Empirical Tables ---

def generate_firm_tables():
    
    for firm in FIRMS:
        # Pre-load data for this firm for all dimensions to avoid opening files inside the row loop
        # Structure: firm_data[m] = { 'lhcc_fil': ..., 'lhcc_kal': ..., 'lhcc_se': ..., 'afc_kal': ..., 'afc_se': ..., 'll': ... }
        firm_data = {}
        
        firm_dir = BASE_PATH + "/" + firm
        
        for m in range(1, 4):
            data_pack = {'exists': False}
            
            # Construct Filepaths
            # NOTE: Logic adjusted based on user description.
            # 1. LHCC Filipovic (No SE)
            fp_fil = firm_dir  + "/" +  f"Filipovic_LHC_NX{m}.npz"
            # 2. LHCC Kalman (Has SE)
            fp_lhc_kal = firm_dir  + "/" +  f"Kalman_resultsLHC_NX{m}.npz"
            # 3. AFC Kalman (Has SE)
            fp_afc = firm_dir  + "/" +  f"Kalman_resultsCIR_Xdim{m}.npz"
            
            # Load LHCC Filipovic
            d = np.load(fp_fil)
            data_pack['lhcc_fil'] = d['final_param'].flatten() # flatten just in case
            
            # Load LHCC Kalman
            d = np.load(fp_lhc_kal)
            data_pack['lhcc_kal'] = d['final_param'].flatten()
            # Update params in lhcc. 
            data_pack['lhcc_se'] = d['SE'].flatten()
            # Overwrite SE to remove SE on gamma1 in outour.
            se = np.array(d['SE'].flatten())
            data_pack['lhcc_se'] = np.concatenate([se[:2*m],np.array([0]), se[2*m:]])

            data_pack['lhcc_ll'] = d['LL'] if 'LL' in d else d.get('log_likeli', None)

            # Load AFC Kalman
            d = np.load(fp_afc)
            data_pack['afc_kal'] = d['final_param'].flatten()
            # Rerun se for afc model -> asburdly large sE likely due to num errors
            sub_df = pd.read_excel("./Data/subset_data.xlsx")

            test_df = sub_df[(sub_df['Ticker']==firm)]
            test_df = test_df.pivot(index = ['Date','Ticker'],
                                    columns='Tenor',values = 'Par Spread').reset_index()
            # Test on subset data ownly to get very few obs. One large spread increase to test.
            test_df['Years']= ((test_df['Date'] - test_df['Date'].min()).dt.total_seconds() / (365.25 * 24 * 3600)).drop_duplicates()

            t = np.array(test_df['Years'])

            mat_grid = np.array([1,2,3,4,5,7,10])
            t_mat_grid = np.ascontiguousarray(mat_grid[:, None] + t[None, :])   # shape (len(T_M_grid), len(t_obs))
            # Forward fill in case of nans.
            CDS_obs = np.array(test_df[['1Y','2Y','3Y','4Y','5Y','7Y','10Y']].ffill().bfill())

            # Read in inferred survival probs.
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

            r = 0.00248
            delta = 0.4
            tenor = 0.25
            cir = CIRIntensity(r,delta,tenor,m,cascading=True)
            # Set new optimal parameters too.
            cir.set_params(data_pack['afc_kal'])

            data_pack['afc_se']  =cir.kalman_SE(data_pack['afc_kal'],
                                                t_obs=t, t_mat_grid=t_mat_grid, 
                                                Y=Gamma_kalman,result=True,eps=1e-5)



            # data_pack['afc_se'] = d['SE'].flatten()
            data_pack['afc_ll'] = d['log_likeli']
            
            data_pack['exists'] = True
            
            firm_data[m] = data_pack

        # --- Start Printing LaTeX Table ---
        print(r"\begin{table}[H]")
        print(r"    \centering")
        print(r"    \resizebox{\textwidth}{!}{") # Optional: Resize to fit if too wide
        print(r"    \begin{tabular}{|c|ccc|ccc|ccc|}")
        print(r"        \hline")
        print(rf"        & \multicolumn{{3}}{{c|}}{{\textbf{{$m=1$}}}} & \multicolumn{{3}}{{c|}}{{\textbf{{$m=2$}}}} & \multicolumn{{3}}{{c|}}{{\textbf{{$m=3$}}}} \\")
        print(r"        \cline{2-10}")
        print(r"        Parameter & LHCC Fil. & LHCC Kal. & AFC & LHCC Fil. & LHCC Kal. & AFC & LHCC Fil. & LHCC Kal. & AFC \\")
        print(r"        \hline")

        # Loop through Parameters
        for param_key, latex_label in ROW_ORDER_FIRMS:
            row_cells = []
            
            for m in range(1, 4):
                d = firm_data.get(m, {})
                
                # 1. LHCC Filipovic (Value only)
                val_fil = get_param_value(param_key, d.get('lhcc_fil'), m,model='lhc_f')
                str_fil = format_val(val_fil)
                str_fil = r"\makecell{" + str_fil + r"\\[-6pt] \small{}}" # Padding for alignment

                # 2. LHCC Kalman (Value + SE)
                val_kal = get_param_value(param_key, d.get('lhcc_kal'), m,model='lhc')
                se_kal = get_param_value(param_key, d.get('lhcc_se'), m,model='lhc')
                
                str_kal = format_val(val_kal)
                str_se_kal = format_val(se_kal)
                
                if str_kal != "-":
                    str_kal = r"\makecell{" + str_kal + r"\\[-6pt] \small{(" + str_se_kal + r")}}"
                else:
                    str_kal = "-"

                # 3. AFC Kalman (Value + SE)
                # Check exclusion: AFC has no Gamma
                if "gamma" in param_key:
                    str_afc = "-"
                else:
                    val_afc = get_param_value(param_key, d.get('afc_kal'), m,model='affine')
                    se_afc = get_param_value(param_key, d.get('afc_se'), m,model='affine')
                    
                    str_afc = format_val(val_afc)
                    str_se_afc = format_val(se_afc)
                    
                    if str_afc != "-":
                        str_afc = r"\makecell{" + str_afc + r"\\[-6pt] \small{(" + str_se_afc + r")}}"
                    else:
                        str_afc = "-"

                row_cells.extend([str_fil, str_kal, str_afc])

            print(f"        {latex_label} & {' & '.join(row_cells)} \\\\")

        print(r"        \hline")
        
        # Log Likelihood Row
        ll_cells = []
        for m in range(1, 4):
            d = firm_data.get(m, {})
            
            # Fil LL (Usually not tracked or same as Kalman, putting '-' based on request or data availability)
            ll_fil = "-" 
            
            # LHCC Kal LL
            val_ll_lhc = d.get('lhcc_ll')
            # Extract single value if it's an array
            if isinstance(val_ll_lhc, (np.ndarray, list)): val_ll_lhc = val_ll_lhc.flat[0]
            str_ll_lhc = format_val(val_ll_lhc)
            
            # AFC Kal LL
            val_ll_afc = d.get('afc_ll')
            if isinstance(val_ll_afc, (np.ndarray, list)): val_ll_afc = val_ll_afc.flat[0]
            str_ll_afc = format_val(val_ll_afc)

            ll_cells.extend([ll_fil, str_ll_lhc, str_ll_afc])

        print(f"        $\\log \\mathcal{{L}}$ & {' & '.join(ll_cells)} \\\\")

        print(r"        \hline")
        print(r"    \end{tabular}}") # End resizebox
        print(f"    \\caption{{Parameter Estimation Results for {firm}}}")
        print(f"    \\label{{tab:results_{firm}}}")
        print(r"\end{table}")
        print("\n\n")

### Function to compare fixing gamma1 relationship and having gamma1 free.
def generate_firm_tables_appendix():
    
    for firm in ['DANBNK','MONTE']:
        # Pre-load data for this firm for all dimensions to avoid opening files inside the row loop
        # Structure: firm_data[m] = { 'lhcc_fil': ..., 'lhcc_kal': ..., 'lhcc_se': ..., 'afc_kal': ..., 'afc_se': ..., 'll': ... }
        firm_data = {}
        
        firm_dir = BASE_PATH + "/" + firm
        
        for m in range(1, 4):
            data_pack = {'exists': False}
            
            # Construct Filepaths
            # NOTE: Logic adjusted based on user description.
            # 2. LHCC Kalman (Has SE)
            fp_lhc_kal = firm_dir  + "/" +  f"Kalman_resultsLHC_NX{m}.npz"
            fp_lhc_kal_gamma1 = firm_dir  + "/" +  f"Kalman_resultsLHC_NX{m}_gamma1.npz"
            
            # Load LHCC Kalman
            d = np.load(fp_lhc_kal)
            data_pack['lhcc_kal'] = d['final_param'].flatten()

            # Overwrite SE to remove SE on gamma1 in outour.
            se = np.array(d['SE'].flatten())
            data_pack['lhcc_se'] = np.concatenate([se[:2*m],np.array([0]), se[2*m:]])

            data_pack['lhcc_ll'] = d['LL'] if 'LL' in d else d.get('log_likeli', None)

            d = np.load(fp_lhc_kal_gamma1)
            data_pack['lhcc_kal_gamma1'] = d['final_param'].flatten()
            # Overwrite SE to remove SE on gamma1 in outour.
            se = np.array(d['SE'].flatten())
            data_pack['lhcc_se_gamma1'] = se

            data_pack['lhcc_ll_gamma1'] = d['LL'] if 'LL' in d else d.get('log_likeli', None)


            data_pack['exists'] = True
            
            firm_data[m] = data_pack

        # --- Start Printing LaTeX Table ---
        print(r"\begin{table}[H]")
        print(r"    \centering")
        print(r"    \resizebox{\textwidth}{!}{") # Optional: Resize to fit if too wide
        print(r"    \begin{tabular}{|c|cc|cc|cc|}")
        print(r"        \hline")
        print(rf"        & \multicolumn{{2}}{{c|}}{{\textbf{{$m=1$}}}} & \multicolumn{{2}}{{c|}}{{\textbf{{$m=2$}}}} & \multicolumn{{2}}{{c|}}{{\textbf{{$m=3$}}}} \\")
        print(r"        \cline{2-7}")
        print(r"        Parameter & Kalman & Kalman w. $\gamma_1$ & Kalman & Kalman w. $\gamma_1$ & Kalman & Kalman w. $\gamma_1$ \\")
        print(r"        \hline")

        # Loop through Parameters
        for param_key, latex_label in ROW_ORDER_FIRMS:
            row_cells = []
            
            for m in range(1, 4):
                d = firm_data.get(m, {})

                # 2. LHCC Kalman (Value + SE)
                val_kal = get_param_value(param_key, d.get('lhcc_kal'), m,model='lhc')
                se_kal = get_param_value(param_key, d.get('lhcc_se'), m,model='lhc')
                
                str_kal = format_val(val_kal)
                str_se_kal = format_val(se_kal)
                
                if str_kal != "-":
                    str_kal = r"\makecell{" + str_kal + r"\\[-6pt] \small{(" + str_se_kal + r")}}"
                else:
                    str_kal = "-"


                # 2. LHCC Kalman (Value + SE)
                val_kal_gamma1 = get_param_value(param_key, d.get('lhcc_kal_gamma1'), m,model='lhc')
                se_kal_gamma1 = get_param_value(param_key, d.get('lhcc_se_gamma1'), m,model='lhc')
                
                str_kal_gamma1 = format_val(val_kal_gamma1)
                str_se_kal_gamma1 = format_val(se_kal_gamma1)
                
                if str_kal_gamma1 != "-":
                    str_kal_gamma1 = r"\makecell{" + str_kal_gamma1 + r"\\[-6pt] \small{(" + str_se_kal_gamma1 + r")}}"
                else:
                    str_kal_gamma1 = "-"

                row_cells.extend([str_kal,str_kal_gamma1])

            print(f"        {latex_label} & {' & '.join(row_cells)} \\\\")

        print(r"        \hline")
        
        # Log Likelihood Row
        ll_cells = []
        for m in range(1, 4):
            d = firm_data.get(m, {})
            
            # LHCC Kal LL
            val_ll_lhc = d.get('lhcc_ll')
            # Extract single value if it's an array
            if isinstance(val_ll_lhc, (np.ndarray, list)): val_ll_lhc = val_ll_lhc.flat[0]
            str_ll_lhc = format_val(val_ll_lhc)
                        # LHCC Kal LL

            val_ll_lhc_gamma1 = d.get('lhcc_ll_gamma1')
            # Extract single value if it's an array
            if isinstance(val_ll_lhc, (np.ndarray, list)): val_ll_lhc_gamma1 = val_ll_lhc_gamma1.flat[0]
            str_ll_lhc_gamma1 = format_val(val_ll_lhc_gamma1)

            ll_cells.extend([str_ll_lhc, str_ll_lhc_gamma1])

        print(f"        $\\log \\mathcal{{L}}$ & {' & '.join(ll_cells)} \\\\")

        print(r"        \hline")
        print(r"    \end{tabular}}") # End resizebox
        print(f"    \\caption{{Parameter Estimation Results for {firm}}}")
        print(f"    \\label{{tab:results_app_{firm}}}")
        print(r"\end{table}")
        print("\n\n")



    #### Function for creating summary table with average, mean min and max CDS spread.

def cds_stats_latex_by_firm(df):
    rows = []
    df = df.pivot(index = ['Date','Ticker'],
                                 columns='Tenor',values = 'Par Spread').reset_index()
    df  = df[df['Ticker'].isin(['DANBNK','MONTE'])]
    frames = []

    for firm, df_sub in df.groupby('Ticker'):

        df_t = df_sub[['1Y','2Y','3Y','4Y','5Y','7Y','10Y']].ffill().bfill()

        stats = pd.DataFrame(
            [
                df_t.mean(),
                df_t.std(),
                df_t.median(),
                df_t.min(),
                df_t.max()
            ],
            index=["Mean", "Std", "Median", "Min", "Max"]
        ) * 10000

        stats.index = pd.MultiIndex.from_product(
            [[firm], stats.index],
            names=["Firm", "Statistic"]
        )

        frames.append(stats)

    out = pd.concat(frames)
    mon_obs = df[df['Ticker'].isin(['MONTE'])].shape[0]* df[df['Ticker'].isin(['MONTE'])].shape[1]
    dan_obs = df[df['Ticker'].isin(['DANBNK'])].shape[0]* df[df['Ticker'].isin(['DANBNK'])].shape[1]
    latex = out.to_latex(
        float_format="%.2f",
        multirow=True,
        caption=f"CDS Spread statistics (bps).  DANBNK {dan_obs} obs. MONTE {mon_obs} obs.",
        label="tab:cds_stats"
    )
    print(latex)

def gamma_mkt_table():
    global_path = './Gamma_Calibration'
    firms = ['DANBNK', 'MONTE']
    tenors = ['1Y', '2Y', '3Y', '4Y', '5Y', '7Y', '10Y']

    gamma_mkt_dict = {}

    for firm in firms:
        data_p = f"{global_path}/{firm}/Data_{firm}.npz"
        
        data = np.load(data_p)
        gamma_mkt = data['gamma_hist']
        
        # first date, all maturities
        gamma_mkt_dict[firm] = gamma_mkt[0, :]

    # Firms as rows, tenors as columns
    out = pd.DataFrame.from_dict(
        gamma_mkt_dict,
        orient='index',
        columns=tenors
    )

    latex = out.to_latex(
        float_format="%.4f",
        caption=r"$\gamma^{mkt}$ on January 1st 2019",
        label="tab:gamma_mkt"
    )

    print(latex)

# def cdso_table():


# def digital_table():

def lookback_table():
    df = pd.DataFrame(
        index=["DANBNK", "MONTE"]
    )
    for x_dim in [1,2,3]:
        look_MC = []
        look_MC_cir = []
        look_MC_g = []
        for firm in ['DANBNK','MONTE']:
            # Read in data.
            
            directory = f"./Results/{firm}"
            filepath = os.path.join(directory, f"Option_data_{firm}_X{x_dim}.npz")
            option_data = np.load(filepath)
            if firm == 'MONTE':
                monte_spot = round(option_data['cds_obs']*10000,3)
            elif firm == 'DANBNK':
                danbnk_spot = round(option_data['cds_obs']*10000,3)
            look_MC.append(round(option_data['look_MC_lhc']*10000,3))
            look_MC_cir.append(round(option_data['look_MC_cir']*10000,3)   )
            filepath = os.path.join(directory, f"Option_data_{firm}_X{x_dim}_full.npz")
            option_data_full = np.load(filepath)
            look_MC_g.append(round(option_data_full['look_MC_lhc']*10000,3))
            # Append to list.
        df[f"$LHCC({x_dim})$"] = look_MC 
        df[r"$LHCC_{\gamma}"+f"({x_dim})$"] = look_MC_g 
        df[f"$AFC({x_dim})$"] = look_MC_cir 
            
    latex = df.to_latex(
    column_format="c|ccc|ccc|ccc",
    float_format="%.2f",
    escape=False,
    caption="Lookback Call Option Prices in Basis Points",
    label="tab:lookback",
    )

    print(latex)
    print("\n")
    print(f'DANBNK spot: {danbnk_spot}, MONTE spot: {monte_spot}')




if __name__ == "__main__":
    print(f'Market calibration gamma')
    gamma_mkt_table()
    print("\n")
    print("% --- Simulation Tables ---")
    generate_latex_lhc()
    print("\n")
    generate_latex_afc()
    
    print("\n% --- Empirical Firm Tables ---")
    generate_firm_tables()

    print("\n% --- Empirical Firm Tables with gamma1 ---")
    generate_firm_tables_appendix()


    # New simulation results:
    files_map = {
    1: "./Simulation_studies/lhc_parameter_comparison_Xdim1_wgamma1.xlsx",
    2: "./Simulation_studies/lhc_parameter_comparison_Xdim2_wgamma1.xlsx",
    3: "./Simulation_studies/lhc_parameter_comparison_Xdim3_wgamma1.xlsx" # Placeholder name
    }
    generate_latex_lhc()
    data = pd.read_excel("./Data/subset_data.xlsx")
    cds_stats_latex_by_firm(data)


    ### OPTION TABLES:
    # Call option table excluding 

    # Lookback
    lookback_table()

test = 1