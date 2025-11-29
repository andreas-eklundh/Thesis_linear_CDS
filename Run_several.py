import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from Models.LHCModels.LHC_single import LHC_single, rebuild_lhc_struct,kalmanfilter_opt,build_P_params
from Models.BaselineCIR_alternative.CIR_Multifactor import CIRIntensity
import os
if __name__ == "__main__":

    t_grid = [0 + 0.25* i for i in range(0, int(10 / 0.25)+1)]

    #### Preliminary investigation.
    sub_df = pd.read_excel("./Data/subset_data.xlsx")
    firms = ['CMZB','DANBNK','MONTE', 'SVSKHB'] # IG,IG,HY,IG
    # Pivot
    # firms = [ 'DANBNK','MONTE', 'SVSKHB'] # IG,IG,HY,IG
    # firms = ['SVSKHB']
## Section 1.1 LHC Filipociv
    for firm in firms:
        test_df = sub_df[(sub_df['Ticker']==firm)]

        test_df = test_df.pivot(index = ['Date','Ticker'],
                                columns='Tenor',values = 'Par Spread').reset_index()

        # Test on subset data ownly to get very few obs. One large spread increase to test.
        test_df['Years']= ((test_df['Date'] - test_df['Date'].min()).dt.total_seconds() / (365.25 * 24 * 3600)).drop_duplicates()

        t = np.array(test_df['Years'])

        mat_grid = np.array([1,3,5,7,10])
        t_mat_grid = np.ascontiguousarray(mat_grid[:, None] + t[None, :])   # shape (len(T_M_grid), len(t_obs))
        # Forward fill in case of nans.
        CDS_obs = np.array(test_df[['1Y','3Y','5Y','7Y','10Y']].ffill().bfill())

        ### Run models. For starters, we are just testing varois kalman versions.
        for X_dim in [1,2,3]:
            lhc = LHC_single( r=0.0252,delta=0.4,cds_tenor= 0.25 )
            lhc.initialise_LHC(Y_dim=1,X_dim=X_dim,X0=0.5,rng=None)


            optim_params= lhc.optimize_params(t, t_mat_grid, CDS_obs)

            # After optimize, define lhc model for inputting.
            # ---- Get states ----
            X, Y, Z = lhc.get_states(t, t_mat_grid, CDS_obs)
            S = Y  # rename Y to S

            default_intensity = lhc.default_intensity(X,Y)

            # ---- Model CDS ----
            CDS_model = lhc.CDS_model(t, t_mat_grid, CDS_obs)

            directory = f"C:/Users/andre/OneDrive/KU, MAT-OEK/Kandidat/Thesis/Thesis_linear_CDS/Results/{firm}"
            filepath = os.path.join(directory, f"Filipovic_LHC_NX{X_dim}.npz")

            # Ensure directory exists
            os.makedirs(directory, exist_ok=True)

            np.savez(filepath,
                    final_param=optim_params,
                    Xn=X,
                    Yn=Y,
                    Default_intensity = default_intensity,
                    CDS_model = CDS_model)
            print(f'Finised X_dim {X_dim}, firm {firm}')


# SECTION 1.2: LHC KALMAN FITS
    # for firm in firms:
    #     test_df = sub_df[(sub_df['Ticker']==firm)]
    #     test_df = test_df.pivot(index = ['Date','Ticker'],
    #                             columns='Tenor',values = 'Par Spread').reset_index()
    #     # Test on subset data ownly to get very few obs. One large spread increase to test.
    #     test_df['Years']= ((test_df['Date'] - test_df['Date'].min()).dt.total_seconds() / (365.25 * 24 * 3600)).drop_duplicates()

    #     t = np.array(test_df['Years'])

    #     mat_grid = np.array([1,3,5,7,10])
    #     t_mat_grid = np.ascontiguousarray(mat_grid[:, None] + t[None, :])   # shape (len(T_M_grid), len(t_obs))
    #     # Forward fill in case of nans.
    #     CDS_obs = np.array(test_df[['1Y','3Y','5Y','7Y','10Y']].ffill().bfill())

    #     ### Run models. For starters, we are just testing varois kalman versions.
    #     for X_dim in [1,2,3]:
    #         lhc = LHC_single( r=0.0252,delta=0.4,cds_tenor= 0.25 )
    #         lhc.initialise_LHC(Y_dim=1,X_dim=X_dim,X0=0.5,rng=None)

    #         optim_params,  Xn,_, Pn, se,ll= lhc.run_n_kalmans(t, t_mat_grid, CDS_obs,base_seed = 300,n_restarts=1)
    #         Xn_kalman,Yn_kalman = lhc.kalman_X_Y(t,Xn)
    #         # Get reconstructed spreads
    #         Zn = lhc.CDS_model(t, t_mat_grid,CDS_obs,t,Xn_kalman.T,Yn_kalman)

    #         print(f'Optimal Paramerters {optim_params}')
    #         kappa, theta, gamma1 = optim_params[:lhc.m],optim_params[lhc.m:2*lhc.m], optim_params[2*lhc.m]

    #         default_intensity = lhc.default_intensity(Xn_kalman.T,Yn_kalman)
    #         directory = f"C:/Users/andre/OneDrive/KU, MAT-OEK/Kandidat/Thesis/Thesis_linear_CDS/Results/{firm}"
    #         filepath = os.path.join(directory, f"Kalman_resultsLHC_NX{X_dim}.npz")

    #         # Ensure directory exists
    #         os.makedirs(directory, exist_ok=True)

    #         np.savez(filepath,
    #                 final_param=optim_params,
    #                 Xn=Xn_kalman,
    #                 Yn=Yn_kalman,
    #                 Pn = Pn,
    #                 Default_intensity = default_intensity,
    #                 CDS_model = Zn,
    #                 SE = se,
    #                 LL = ll) #,
    #         print(optim_params)
    #         print(f'Finised X_dim {X_dim}, firm {firm}')
        # Then run comparison plots a la result_plots.
        # But run in seperate loop so no need for rerun.

##### SECTION 2: CIR MODEL FITS.
    # for firm in firms:

    #     r = 0.0252
    #     delta = 0.4
    #     tenor = 0.25

    #     test_df = sub_df[(sub_df['Ticker']==firm)]
    #     test_df = test_df.pivot(index = ['Date','Ticker'],
    #                             columns='Tenor',values = 'Par Spread').reset_index()
    #     # Test on subset data ownly to get very few obs. One large spread increase to test.
    #     test_df['Years']= ((test_df['Date'] - test_df['Date'].min()).dt.total_seconds() / (365.25 * 24 * 3600)).drop_duplicates()

    #     t = np.array(test_df['Years'])

    #     mat_grid = np.array([1,3,5,7,10])
    #     t_mat_grid = np.ascontiguousarray(mat_grid[:, None] + t[None, :])   # shape (len(T_M_grid), len(t_obs))
    #     # Forward fill in case of nans.
    #     CDS_obs = np.array(test_df[['1Y','3Y','5Y','7Y','10Y']].ffill().bfill())

    #     # Read in inferred survival probs.
    #     data = np.load(f"C:/Users/andre/OneDrive/KU, MAT-OEK/Kandidat/Thesis/Thesis_linear_CDS/Gamma_Calibration/{firm}/Data_{firm}.npz")
    #     t_mats_plots = data['t_mats_plots']
    #     survival=data['survival']
    #     Gamma = data['Gamma']
    #     default_prob = data['default_prob']

    #     t_mats_plots_kalman = t_mats_plots[np.isin(t_mats_plots,mat_grid).flatten()]

    #     survival_kalman = survival[:,np.isin(t_mats_plots, mat_grid).flatten()]
    #     Gamma_kalman = Gamma[:,np.isin(t_mats_plots, mat_grid).flatten()]
    #     Gamma_kalman_scale =Gamma_kalman / mat_grid[None, :]

    #     # Negative process. Multiply by -1 everywhere. Let A and a get these too.
    #     for X_dim in [2,3]:# , 2]:
    #         cir = CIRIntensity(r,delta,tenor,X_dim,cascading=True)
    #         x0 = np.array([0])
    #         ll, params, Xn,Zn,Pn,se = cir.run_kalman_filter(t,t_mat_grid,
    #                                                     Y=Gamma_kalman_scale ,seed=2000)


    #         # Set new optimal parameters too.
    #         cir.set_params(params)
    #         # Save values:

    #         # With Params in place, we can utilize CIR class to do pricing, simulations etc.
    #         CDS_cir = np.zeros(survival_kalman.shape)
    #         for n in range(survival_kalman[:,0].shape[0]):
    #             CDS_cir[n,:] = cir.cds_spread_fast(Xn[n,:],params,t[n],t_mat_grid[:,n])
    #             print(f'Done with {(n+1)/survival_kalman[:,0].shape[0]} %')

    #         # Default intensity probability since inception (identical to state if dim=1)
    #         # default_intensity = np.sum(Xn,axis=1)
    #         default_intensity = Xn[:,0]

    #         # Get induced default probability:
    #         Yn = np.exp(-np.cumsum(default_intensity*(t[1]-t[0]))) # only approximates

    #         # Actual/observed

    #         np.savez(f"C:/Users/andre/OneDrive/KU, MAT-OEK/Kandidat/Thesis/Thesis_linear_CDS/Results/{firm}/Kalman_resultsCIR_Xdim{X_dim}.npz",
    #                 final_param=params,
    #                 Xn=Xn,
    #                 Zn=Zn,
    #                 Pn = Pn,
    #                 Yn = Yn,
    #                 default_intensity = default_intensity,
    #                 CDS_cir = CDS_cir,
    #                 SE = se,
    #                 log_likeli = ll)



