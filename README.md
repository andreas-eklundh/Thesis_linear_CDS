# Thesis_linear_CDS

This note descibes the main content of the project, and what code to run in order to reproduce findings. 

## Main body
There are four main classes all located under models. 
* LHC_singly.py. This is the LHCC class holding functions and methods for LHCC calibration
* CIR_Multifactor.py. Class with the affine benchmark model
* moments: Implementation of theorem 2 in the thesis i.e. the Polynomial conditional moments. 
* Gamma_solver.py: Class for converting observed spreads into deterministic default intensities. 
* ATSM.py: The general affine class as of D. Duffie. 

Other scripts and methods are used and utilized in the above. However, this is where the modeling work is. 

## Output scripts
 
* Data_prep.py: Cleans data. 
* Simulation_studies.py: Runs the simulations of the affine class and lhc model with known parameters.
* Likelihoods.py: Outputs likelihood plots of chapter 6 using values above. 
* Convert_spreads.py. Utilizes the Gamma_solver to convert spreads to deterministic default intensities. 
* Run_several.py: Runs the models for MONTE, DANBNK.
* Out_several.py: outputs figures in text for run_several.py. These figures are used in the thesis.
* Out_options.py: Sends out option prices as described in the thesis.
* Format results.py: This takes some of the created data obtained by running the above and outputs it the tables used in the thesis. 
* PCA_idea_generation.py: This runs a very small PCA analysis. 


Other files have primarily been added for testing various thoughts, ideas etc. during the progress of the thesis. 





