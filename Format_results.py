import pandas as pd
import numpy as np
import re

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
excel_files = {
    1: "./Simulation_studies/lhc_parameter_comparison_Xdim1.xlsx",
    2: "./Simulation_studies/lhc_parameter_comparison_Xdim2.xlsx",
    3: "./Simulation_studies/lhc_parameter_comparison_Xdim3.xlsx"
}

# The parameters and ordering EXACTLY as in your LaTeX template
ordered_params = [
    "kappa1","kappa2","kappa3",
    "theta1","theta2","theta3",
    "gamma1",
    "lambda1","lambda2","lambda3",
    "sigma1","sigma2","sigma3",
    "sigma_err"
]

# Map internal names → LaTeX math
param_to_latex = {
    "kappa1": r"$\kappa_1$",
    "kappa2": r"$\kappa_2$",
    "kappa3": r"$\kappa_3$",
    "theta1": r"$\theta_1$",
    "theta2": r"$\theta_2$",
    "theta3": r"$\theta_3$",
    "gamma1": r"$\gamma_1$",
    "lambda1": r"$\lambda_1$",
    "lambda2": r"$\lambda_2$",
    "lambda3": r"$\lambda_3$",
    "sigma1": r"$\sigma_1$",
    "sigma2": r"$\sigma_2$",
    "sigma3": r"$\sigma_3$",
    "sigma_err": r"$\sigma_{err}$"
}

# ---------------------------------------------------------
# FUNCTION TO CLEAN PARAMETER NAME
# ---------------------------------------------------------
def normalize(p):
    p = p.lower().replace(" ", "")
    p = p.replace("sigma_err","sigma_err")
    p = p.replace("sigmaerror","sigma_err")
    p = p.replace("lambda","lambda")
    m = re.match(r"([a-zA-Z]+)(\d*)", p)
    if m:
        base, num = m.groups()
        return base + num
    return p

# ---------------------------------------------------------
# READ AND ALIGN FILES
# ---------------------------------------------------------
data = {}

for model, file in excel_files.items():
    df = pd.read_excel(file)
    df.columns = df.columns.str.lower().str.strip()

    # standardize names
    df["param"] = df[df.columns[0]].apply(lambda x: normalize(str(x)))

    keep = ["param"] + [c for c in df.columns if any(k in c for k in ["True",
                                                                      "estimated kalman",
                                                                      "se",
                                                                      "filipovic"])]
    df = df[keep]

    df.set_index("param", inplace=True)

    data[model] = df

# ---------------------------------------------------------
# LATEX CELL BUILDER
# ---------------------------------------------------------
def latex_cell(true, est, se, filip):
    # convert missing values to "-"
    def fmt(x):
        if pd.isna(x): return "-"
        if isinstance(x,str): return x
        return f"{x:.4g}"

    true = fmt(true)
    fil = fmt(filip)

    # Kalman + SE stacked
    if pd.isna(est):
        kalman_cell = "-"
    else:
        est_str = fmt(est)
        if pd.isna(se):
            kalman_cell = est_str
        else:
            se_str = fmt(se)
            kalman_cell = f"{est_str}\\\\ {{\\scriptsize ({se_str})}}"

    return true, kalman_cell, fil

# ---------------------------------------------------------
# BUILD LATEX OUTPUT
# ---------------------------------------------------------
latex = []
latex.append(r"\begin{table}[H]")
latex.append(r"    \centering")
latex.append(r"    \begin{tabular}{|c|ccc|ccc|ccc|}")
latex.append(r"        \hline")
latex.append(r"        Parameters & \multicolumn{3}{c|}{LHCC(1)} & \multicolumn{3}{c|}{LHCC(2)} & \multicolumn{3}{c|}{LHCC(3)} \\")
latex.append(r"        \cline{2-10}")
latex.append(r"         & True & Kalman & Filipovic & True & Kalman & Filipovic & True & Kalman & Filipovic \\")
latex.append(r"        \hline")

for p in ordered_params:
    row = [param_to_latex[p]]
    for m in [1,2,3]:
        df = data[m]

        t = df.at[p,"true"] if "true" in df.columns and p in df.index else np.nan
        e = df.at[p,"est"] if "estimkalman" in df.columns and p in df.index else np.nan
        s = df.at[p,"se"] if "se" in df.columns and p in df.index else np.nan
        f = df.at[p,"filip"] if "filip" in df.columns and p in df.index else np.nan

        t, k, f = latex_cell(t,e,s,f)
        row += [t,k,f]

    latex.append("         " + " & ".join(row) + r" \\")
    
latex.append(r"        \hline")
latex.append(r"    \end{tabular}")
latex.append(r"    \caption{Parameter Estimation in the LHCC Model. Standard Errors in Parenthesis}")
latex.append(r"    \label{tab:lhc_simul}")
latex.append(r"\end{table}")

# Print result
print("\n".join(latex))


test = 1