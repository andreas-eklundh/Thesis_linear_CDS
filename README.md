# Thesis_linear_CDS

This note descibes the main content of the project, and what code to run in order to reproduce findings. 

## Main body
There are four main classes all located under models. 
* LHC_singly.py. This is the LHCC class holding functions and methods for LHCC calibration
* CIR_Multifactor.py. Class with the affine benchmark model
* moments: Implementation of theorem 2 in the thesis i.e. the Polynomial conditional moments. 
* Gamma_solver.py: Class for converting observed spreads into deterministic default intensities. 
* ATSM.py: The general affine class as of Duffie. 

Other scripts are used and utilized in the above. However, this is where the modeling work is. 

## Output scripts

* Simulation_studies.py: Runs the simulations of the affine class and lhc model with known parameters.
* Run_several.py: Runs the models for MONTE, SVSKHB, DANBNK, CZMB. 
* Data_prep.py: Cleans data. 
* Out_several.py: outputs figures in text for run_several.py.
* Out_options.py: Sends out option prices as described in the thesis.
* Convert_spreads.py. Utilizes the Gamma_solver to convert spreads to deterministic default intensities. 




