import pandas as pd
import os
import numpy as np

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
FIRMS = ['CMZB', 'DANBNK', 'MONTE', 'SVSKHB']
BASE_PATH = r"C:/Users/andre/OneDrive/KU, MAT-OEK/Kandidat/Thesis/Thesis_linear_CDS/Results"

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
        if abs(v) < 1e-4:
            return f"{v:.2e}"   # scientific format
        
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
            # If the file is missing or the specific parameter isn't in this dimension
            # We output "-" if the file exists but param is missing, or empty if file is missing?
            # Based on your table, empty entries (no hyphen) usually mean "not applicable" for that dimension.
            # While "-" means "applicable but missing/zero".
            # For now, if the DF is missing (Dim 3), we leave it blank. 
            # If DF exists but param missing (e.g. kappa2 in Xdim1), we usually leave it blank or put "-".
            # The prompt has empty slots for k2 in LHCC(1).
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

def generate_latex_afc():
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

    # --- LOGIC TO MAP NAME TO ARRAY INDEX ---
    # Adjust these offsets based on your specific vector construction!
    
    # 0-based index conversion
        idx_in_group = p_idx - 1 
    
    # Example logic (ADJUST THIS):
    # Array = [k1..km, th1..thm, sig1..sigm, ...]
    # Logic if Affine
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
            # Update params in lhcc. Apparently lambda is last (my bad)s
            data_pack['lhcc_se'] = d['SE'].flatten()
            data_pack['lhcc_ll'] = d['LL'] if 'LL' in d else d.get('log_likeli', None)

            # Load AFC Kalman
            d = np.load(fp_afc)
            data_pack['afc_kal'] = d['final_param'].flatten()
            data_pack['afc_se'] = d['SE'].flatten()
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

if __name__ == "__main__":
    print("% --- Simulation Tables ---")
    generate_latex_lhc()
    print("\n")
    generate_latex_afc()
    
    print("\n% --- Empirical Firm Tables ---")
    generate_firm_tables()


test = 1