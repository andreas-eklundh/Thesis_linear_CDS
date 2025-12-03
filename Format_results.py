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



if __name__ == "__main__":
    ### The LHCC model
    generate_latex_lhc()


    ### The Cascading affine model-
    generate_latex_afc()



#### New section for handling the npz files and convert these into output tables. 
# Logic for handling the tables. 



# Loop over firms when in prod.
firms = ['CZMB','DANBNK', 'MONTE', 'SVSKHB']


for firm in firms:
    # Stack multicol wise.
    for m in range(1,4):
        ### Read in data. 

        # LHCC Filipovic data

        directory = f"C:/Users/andre/OneDrive/KU, MAT-OEK/Kandidat/Thesis/Thesis_linear_CDS/Results/{firm}"

        filepath = os.path.join(directory, f"Kalman_resultsLHC_NX{m}.npz")
        data = np.load(filepath)
        filipovic = data["final_param"]

        # LHCC Kalman

        filepath = os.path.join(directory, f"Filipovic_LHC_NX{m}.npz")
        data = np.load(filepath)
        LHCCKalman =data["final_param"]
        se_lhcc = data['SE']
        ll_lhcc = data['LL']
        # AFC Kalman.

        filepath = os.path.join(directory, f"Kalman_resultsCIR_Xdim{m}.npz")
        data = np.load(filepath)
        AFC = data['final_param']
        se_acf = data['SE']
        ll_acf = data['log_likeli']
        # Then fill multicols iteratively.






test = 1