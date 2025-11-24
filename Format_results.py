### Script for formatting tables for latex etc. 
### Generally, other scripts compares model, here we print results alongside.

import numpy as np
import re
import pandas as pd
import os



def format_scientific_latex(number, precision=3):
    """Converts a number to a LaTeX-formatted string in scientific notation with N significant figures."""
    if not isinstance(number, (int, float, np.floating)):
        return str(number)
    
    # Use f-string to enforce 3 significant figures
    s = f'{number:.{precision}g}'
    
    if 'e' in s or 'E' in s:
        # Split into mantissa and exponent
        mantissa, exponent = re.split('e|E', s)
        
        # Ensure mantissa has no trailing '.0' and is stripped of unnecessary zeros
        mantissa = re.sub(r'\.?0+$', '', mantissa)

        # LaTeX format: 1.23 \times 10^{4}
        return f'${mantissa} \\cdot 10^{{{int(exponent)}}}$'
    
    # If not scientific, ensure it's still rounded to 3 significant figures for consistency
    return f'{s}'


##### Formatting for Simulation studies
##### Tables with actual and retrieved parameters.
# lhc1 = pd.read_excel("./Simulation_studies/lhc_parameter_comparison_Xdim1.xlsx")
# lhc2 = pd.read_excel("./Simulation_studies/lhc_parameter_comparison_Xdim2.xlsx")
# lhc3 = pd.read_excel("./Simulation_studies/lhc_parameter_comparison_Xdim3.xlsx")

FILE_PATHS = {
    'LHCC(1)': "./Simulation_studies/lhc_parameter_comparison_Xdim1.xlsx",
    'LHCC(2)': "./Simulation_studies/lhc_parameter_comparison_Xdim2.xlsx",
    'LHCC(3)': "./Simulation_studies/lhc_parameter_comparison_Xdim3.xlsx",
}

# List to hold the DataFrames after processing
dfs_to_combine = []

# --- 2. READ AND RESTRUCTURE EACH FILE ---

for model_name, path in FILE_PATHS.items():
    # Read the file. Assuming the first column is the parameter name (index)
    df_read = pd.read_excel(path, index_col=0)
    # Subset 
    df_read = df_read[['Estimated Kalman','True']]
    # Clean up the index names (e.g., remove 'gamma1' and use 'gamma_1' for consistent mapping)
    df_read.index = df_read.index.str.replace(r'(\w)(\d+)', r'\1_\2', regex=True).str.replace('err', 'err_')
    # Extract SE rows
    SE = df_read.loc[df_read.index.str.contains('SE'), 'Estimated Kalman'].copy()
    SE.index = SE.index.str.replace(r'^.*?\((.*?)\)$', r'\1', regex=True)  # convert SE(kappa1) → kappa1

    # Keep only parameter rows
    df_read = df_read[~df_read.index.str.contains('SE')]

    # Insert SE under Est. inside the same cell (LaTeX)
    def combine_est_se(est, se):
        est = format_scientific_latex(est)
        se = format_scientific_latex(se)
        

        """Return a LaTeX makecell with estimate on top and SE below."""
        if est in ['-', 'nan'] or se in ['-', 'nan']:
            return est  # no SE available (e.g., dimension missing)

        return (
            r"\makecell{" + f"{est}\\\\"
            r"{\scriptsize (" + f"{se}" + r")}}"
        )
    df_read['Est_with_SE'] = [
        combine_est_se(df_read['Estimated Kalman'].iloc[i], SE.get(idx, np.nan))
        for i, idx in enumerate(df_read.index)
    ]

    df_read['Estimated Kalman'] = df_read['Est_with_SE']
    df_read.drop(columns=['Est_with_SE'], inplace=True)

    # Add the model name as the top level of the column MultiIndex
    df_read.columns = pd.MultiIndex.from_product([[model_name], df_read.columns])


    dfs_to_combine.append(df_read)# --- 2. FORMATTING AND INDEX MAPPING ---


df_combined = pd.concat(dfs_to_combine, axis=1)

# 3B. Define the FINAL desired row order (This ensures factors are grouped)
final_index_names = [
    'kappa_1', 'kappa_2', 'kappa_3', 
    'theta_1', 'theta_2', 'theta_3', 
    'gamma_1', # Assuming only one residual gamma/volatility
    'lambda_1', 'lambda_2', 'lambda_3', 
    'sigma_1', 'sigma_2', 'sigma_3', 
    'sigma_err_',
]
# Reindex the DataFrame to match the desired order, adding NaNs where factors are missing
df_final = df_combined.reindex(final_index_names)

# 2A. Map index names to LaTeX symbols
index_map = {
    'kappa_1': r'$\kappa_1$', 'kappa_2': r'$\kappa_2$', 'kappa_3': r'$\kappa_3$',
    'theta_1': r'$\theta_1$', 'theta_2': r'$\theta_2$', 'theta_3': r'$\theta_3$',
    'gamma_1': r'$\gamma_1$',
    'lambda_1': r'$\lambda_1$', 'lambda_2': r'$\lambda_2$', 'lambda_3': r'$\lambda_3$',
    'sigma_1': r'$\sigma_1$', 'sigma_2': r'$\sigma_2$', 'sigma_3': r'$\sigma_3$',
    'sigma_err_': r'$\sigma_{err}$',
}
df_final.index = df_final.index.map(index_map)


# Apply the LaTeX scientific formatting function to all columns
df_final = df_final.applymap(format_scientific_latex)

# 2B. Handle missing values (for factors not present in 1D or 2D models)
df_final.replace('nan', '-', inplace=True) # Replace NaNs (which became 'nan' strings) with '-'

# --- 3. GENERATE LATEX CODE ---
# Generate the table content, excluding all pandas formatting (header, footer, index=False)
latex_table_content = df_final.to_latex(
    escape=False,
    index=True, # Keep the parameter column names in the output
    header=False, # Exclude the column headers (True, Est., Std.)
)

# 3. Clean up the pandas output (remove \toprule, \bottomrule)
latex_table_content = latex_table_content.replace('\\toprule', '').replace('\\bottomrule', '').strip()

# --- 4. ASSEMBLE THE REQUESTED BODY ---

# The body starts with \midrule and ends just before the log-likelihood row.
log_L_row = '\n' + r'\hline ' + r'\log \mathcal{L}' + r' & \multicolumn{3}{c|}{\textbf{CIR(1)}} & \multicolumn{3}{c|}{\textbf{CIR(2)}} & \multicolumn{3}{c|}{\textbf{CIR(3)}} \\'

# The final output is the body rows, followed by the log-likelihood row.
final_body_output = r'\midrule' + '\n' + latex_table_content + log_L_row

print(final_body_output)
test = 1