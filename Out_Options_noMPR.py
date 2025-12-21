
##### Plots for comparison. TODO: Move to other file.
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from Models.LHCModels.LHC_single import LHC_single, get_CDS_Model, rebuild_lhc_struct,kalmanfilter_opt,build_P_params
from Models.BaselineCIR_alternative.CIR_Multifactor import CIRIntensity
import os
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



full_global_summary = []
if __name__ == "__main__":

    t_grid = [0 + 0.25* i for i in range(0, int(10 / 0.25)+1)]

    #### Preliminary investigation.
    sub_df = pd.read_excel("./Data/subset_data.xlsx")
    # firms = ['DANBNK','MONTE','CMZB', 'SVSKHB'] # IG,IG,HY,IG
    # firms = ['DANBNK','MONTE', 'SVSKHB']#, 'SVSKHB'] # IG,IG,HY,IG
    firms = ['DANBNK','MONTE'] # IG,IG,HY,IG

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
    
    X_dims = [1,2,3] # [1,2,3]
    # Run 1-2 first. 3 might be very slow. 
    # X_dims = [1,2,3]
    # for firm in firms:
    #     test_df = sub_df[(sub_df['Ticker']==firm)]
    #     test_df = test_df.pivot(index = ['Date','Ticker'],
    #                             columns='Tenor',values = 'Par Spread').reset_index()
    #     # Test on subset data ownly to get very few obs. One large spread increase to test.
    #     test_df['Years']= ((test_df['Date'] - test_df['Date'].min()).dt.total_seconds() / (365.25 * 24 * 3600)).drop_duplicates()

    #     t = np.array(test_df['Years'])

    #     mat_grid = np.array([1,3,5,7,10])
    #     t_mat_grid = np.ascontiguousarray(mat_grid[:, None] + t[None, :])   # shape (len(T_M_grid), len(t_obs))

    #     # Forwrard fill again. Back fill in case any initial missing
    #     CDS_obs = np.array(test_df[['1Y','3Y','5Y','7Y','10Y']].ffill().bfill())

    #     directory = f"./Results_noMPR/{firm}"
    #     for i in X_dims:
    #         filepath = os.path.join(directory, f"Kalman_resultsLHC_NX{i}.npz")
    #         data = np.load(filepath)
    #         final_paramLHCK = data["final_param"]
    #         XnLHCK = data["Xn"]
    #         YnLHCK=data["Yn"]
    #         PnLHC = data["Pn"]
    #         ZnLHCK = data["CDS_model"]
    #         Default_intensityLHCK = data["Default_intensity"]
    #         print_model_params("LHC Kalman", final_paramLHCK, m=i)


    #         filepath = os.path.join(directory, f"Kalman_resultsCIR_Xdim{i}.npz")
    #         data = np.load(filepath)
    #         final_paramCIR=data['final_param']
    #         XnCIR=data['Xn']
    #         ZnCIR=data['Zn']
    #         PnCIR = data['Pn']
    #         YnCIR = data['Yn']
    #         default_intensityCIR = data['default_intensity']
    #         CDS_cirCIR = data['CDS_cir']
    #         print_model_params("CIR", final_paramCIR, m=i)

    #         ######## Option Pricing in the Models. Here, we can practically only use Kalman-like models
    #         r,delta,tenor = 0.00248,0.4, 0.25
    #         # Initalise LHC model for pricing.
    #         lhc = LHC_single(r,delta,tenor)
    #         Y_dim,X_dim = 1,i
    #         X_now = XnLHCK[-1,:] # Want to predict forward in time.
    #         lhc.initialise_LHC(Y_dim,X_dim,X0=X_now)
    #         lhc.flatten_params()
    #         # Set with optimal parameters
    #         lhc.unflatten_params(final_paramLHCK[:2*X_dim+1])
    #         # Set also P params.
    #         gamma1 = np.array([final_paramLHCK[2*X_dim]])
    #         lhc_P = lhc.build_P_params(params=final_paramLHCK[2*X_dim+1:],gamma1=gamma1)
    #         # Then, we are ready to price in every dimension.
    #         # Strike grid. Base around forward spread
    #         # We consider this setup
    #         t_now= 0.0
    #         t0 = 1.0 # Maturity of CDSoption.
    #         t_mat = t0 + 5.0 # So CDS with 5 yr mat.
    #         cds0_lhc = lhc.CDS_model(t_obs = np.array([t_now]),T_M_grid=np.array([[t_mat-t0]]),
    #                             CDS_obs=None,t0=np.array([t_now]),X_in = XnLHCK[-1,:].reshape((X_dim,1)),
    #                             Y_in = np.array([1]).reshape((1,1)))

    #         ## The inital value at en of data is just observed CDS spread.
    #         cds0 = CDS_obs[-1,2]

    #         # Then strikes around the above. Say PM 150 Bps.
    #         # strike_diff_grid = np.linspace(-60,60,20) / 10000
    #         strike_diff_grid = np.linspace(-40,120,int((120+40)/20)+1) / 10000
    #         strikes = (cds0 + strike_diff_grid).flatten()
    #         # Set Simulate option sizes.
    #         chi0 = np.concatenate((np.array([1]), XnLHCK[-1,:]))
    #         # Set discretization and number of simul. 1m simuls, fairly coarse grid
    #         # N, M = 10000, 100
    #         N, M = 40,100
    #         seed = 2000
    #         # Take Legendre poly at 20. 
    #         leg_deg = 8  # 20 . This is were it explodes in X_dim=3

    #         ### Get CDSO using theoretical values.
    #         # Note, we need to discound too, see top vof p. 19
    #         # To check if actually worth running many. 
    #         # cdso_actual = lhc.get_cdso_price(t_now,t0,t_mat,chi0[0],chi0[1:],np.array([0.005]),n_max=9)
    #         # cdso_MC_hist,cdso_MC = lhc.get_cdso_price_MC(t_now,t0,t_mat,np.array([0.005]),chi0,N,M,seed=1000) 
    #         # print(f'Test: Actual CDSO price {cdso_actual}')
    #         # print(f'Test: MC CDSO price {cdso_MC}')
    #         # Must assume it to be correct. 
    #         cdso_actual = lhc.get_cdso_price(t_now,t0,t_mat,chi0[0],chi0[1:],strikes,n_max=leg_deg)
    #         cdso_MC_hist,cdso_MC = lhc.get_cdso_price_MC(t_now,t0,t_mat,strikes,chi0,N,M,seed=seed)
    #         # N, M = 100,500

    #         ######### DIGITAL OPTION PRICING. ##############
    #         # Define Barriers as some percentage of the CDS at current timepoint.
    #         barriers_percentage = np.array([(1+i*0.05) for i in range(1,11)])
    #         barriers = cds0*barriers_percentage
    #         T_mat_barrier = t_now + (t0-t_now)/2 # Matures halfway to the CDS
    #         digital_MC_hist,digital_MC = lhc.get_digital_barrier_price_MC(t_now,t0,t_mat,T_mat_barrier,barriers,
    #                                                                     chi0,N,M,seed=seed)


    #         ######### Lookback OPTION PRICING. ##############
    #         # Consider CDS at expiry.
    #         look_MC_hist,look_MC,look_cds_min =  lhc.get_lookback_price_MC(t_now,t0,t_mat,T_mat_barrier,
    #                                                                     chi0,N,M,seed=seed)



    #         ###############################################################################
    #         ###############################################################################
    #         ##################### CIR MODELL SIMULATIONS ##################################
    #         ###############################################################################
    #         ###############################################################################



    #         ### CDSO pricing in CIR Model.
    #         cir = CIRIntensity(r,delta,tenor,X_dim,cascading=True)
    #         X_cir_now = XnCIR[-1,:] # Want to predict forward in time.
    #         cir.set_params(final_paramCIR)

    #         # Then, we are ready to price in every dimension.
    #         # Strike grid. Base around forward spread
    #         # We consider this setup
    #         # Get spot CDS rate
    #     #     cds0_CIR = cir.calc_CDS(final_paramCIR,t_now,t_mat-t0,X_cir_now,t0=t_now)

    #     #     print(f"CIR Spread {cds0_CIR}, LHC spread {cds0_lhc}, Observed {cds0}")
    #     #     # For 1-d compare fast vs actual
    #     #     cds0_CIR_n = cir.cds_spread_fast(X_cir_now,final_paramCIR,t_now,np.array([t_mat-t0]),t0=t_now)
    #     #     # Seems about right.
    #     #     print(f'CIR Spread {cds0_CIR}, CIR Spread fast {cds0_CIR_n}')

    #         ### Plot 1 trajectory.
    #         # T_return,lambda_mil_Q_cir = cir.simulate_intensity(lambda0=X_cir_now,
    #         #                                                    T=T_mat_barrier,M=M,
    #         #                                                    scheme="Milstein",
    #         #                                                    seed=seed,measure='Q')
    #         # T_return,lambda_mil_Q_lhc = lhc.simul_latent_states(chi0=chi0,T=T_mat_barrier,
    #         #                                                     M=M,n_mat=1,scheme="Milstein",
    #         #                                                     seed=seed,measure='Q')
    #         # # plt.plot(T_return,lambda_mil_Q_cir[:,1:],color = 'red', label = 'cir' )
    #         # # plt.plot(T_return,lambda_mil_Q_lhc[:,1:],color = 'blue', label = 'lhcc' )
    #         # # plt.legend()
    #         # # plt.show()
            

    #         # # See trajectories of cds spread:
    #         # T_max = t_mat - t[0]
    #         # interp_struct =  cir.precompute_affine_grid(final_paramCIR, T_max, N_grid= 10000+1, rho=1)

            
    #         # # Get model implies CDS.
    #         # CDS_sim = np.zeros(lambda_mil_Q_cir[:,0].flatten().shape)
    #         # for n in range(CDS_sim.shape[0]):
    #         #     CDS_sim[n]  = cir.calc_CDS_fast_numba(T_return[n],t0, t_mat, lambda_mil_Q_cir[n,:],
    #         #                                     interp_struct)
    #         # T_M_grid = (np.array([t_mat]*T_return.shape[0])).reshape((1,T_return.shape[0]))
    #         # t0_arr = (np.array([t0]*T_return.shape[0])).reshape((T_return.shape[0]))
    #         # lhc_n = rebuild_lhc_struct(lhc.kappa, lhc.theta, lhc.gamma1[0], lhc.r,
    #         #                 lhc.Y_dim, lhc.delta, lhc.tenor)
    #         # CDS_sim_lhc = get_CDS_Model(T_return, t0_arr, T_M_grid, lambda_mil_Q_lhc.T, lhc_n )
    #         # plt.plot(T_return,CDS_sim,color = 'red', label = 'cir' )
    #         # plt.plot(T_return,CDS_sim_lhc.T,color = 'blue', label = 'lhcc' )
    #         # plt.legend()
    #         # plt.show()


    #         # Then strikes around the above. Say PM 150 Bps.
    #         # Set discretization and number of simul
    #         cdso_MC_hist_cir,cdso_MC_cir = cir.get_cdso_pric_MC(final_paramCIR,t_now,t0,t_mat,
    #                                                             strikes,XnCIR[-1,:],N,M,seed=seed)

    #         digital_MC_hist_cir,digital_MC_cir = cir.get_digital_barrier_price_MC(final_paramCIR,t_now,t0,t_mat,T_mat_barrier,barriers,
    #                                                                     XnCIR[-1,:],N,M,seed=seed)



    #         ######### Lookback OPTION PRICING. ##############
    #         # Consider CDS at expiry.
    #         look_MC_hist_cir,look_MC_cir,look_cds_min_cir =  cir.get_lookback_price_MC(final_paramCIR,t_now,t0,t_mat,T_mat_barrier,
    #                                                                     XnCIR[-1,:],N,M,seed=seed)

    #         print(f'Lookback price LHC: {look_MC*10000}')
    #         print(f'Lookback price CIR: {look_MC_cir*10000}')


    #         #### Note, in this instance, we actually can compute option prices given estimates states
    #         ### This approach relies on G-transform.
    #         ### Note, we can compute forwards spreads, and based on those price. But not the same as above per se.
    #         # Not really possible unless some change of measure.

    #         # cdso_fourier = np.zeros(strikes.shape[0])

    #         # for j,strike in enumerate(strikes):
    #         #     first_term = cir.G_transform(
    #         #                                 y=-np.log(strike),
    #         #                                 a=np.array([1,0]),
    #         #                                 b=np.array([-1,0]),
    #         #                                 Xt=X_cir_now,
    #         #                                 T=np.array([t0 ])
    #         #                                 )
    #         #     second_term = strike*cir.G_transform(
    #         #                                         y=-np.log(strike),
    #         #                                         a=np.array([0,0]),
    #         #                                         b=np.array([-1,0]),
    #         #                                         Xt=X_cir_now,
    #         #                                         T=np.array([t0 ])
    #         #                                         )
    #         #     cdso_fourier[j] = first_term - second_term



    #         ## Plotting. Use functionality.
    #         # --- convert to basis points ---
    #         strike_offsets_bps = (strike_diff_grid * 1e4).flatten()     # offsets in bps
    #         strikes_bps = (strikes * 1e4).flatten()                     # total strikes in bps
    #         cdso_MC_bps = cdso_MC * 1e4
    #         cdso_MC_hist_bps = cdso_MC_hist * 1e4
    #         cdso_MC_bps_cir = cdso_MC_cir * 1e4
    #         cdso_MC_hist_bps_cir = cdso_MC_hist_cir * 1e4
    #         # cdso_MC_bps_cir_fourier = cdso_fourier * 1e4
        
    #         save_path = f"./Results_noMPR/Options"

    #     #     save_path = directory + f"/Options"
            
    #         os.makedirs(save_path, exist_ok=True)

    #         # --- Plot 1: Price vs Strike Offset ---
    #         fig1, ax1 = plt.subplots(figsize=(8, 6))

    #         ax1.plot(strike_offsets_bps, cdso_MC_bps, 'o-', color='navy', alpha=0.8,label='LHC Model, Simulated')
    #         ax1.plot(strike_offsets_bps, cdso_actual*10000, 'o-', color='red', alpha=0.8,label=f'LHC Model, Legendre n={leg_deg}')
    #         ax1.plot(strike_offsets_bps, cdso_MC_bps_cir, 'o-', color='forestgreen', alpha=0.8,label='CIR Model, Simulated')
    #         # ax1.plot(strike_offsets_bps, cdso_MC_bps_cir_fourier, 'o-', color='grey', alpha=0.8,label='CIR Model, Fourier')

    #         ax1.set_xlabel("Strike Offset (bps)")
    #         ax1.set_ylabel("CDS Option Price (bps)")
    #         # ax1.set_title("CDS Option Price vs Strike Offset")
    #         ax1.grid(True)
    #         ax1.legend(fontsize=8, loc='best')

    #         fig1.tight_layout()
    #         fig1.savefig(os.path.join(save_path, f"CDSO_MC_vs_Strike_{firm}_m{i}.png"), dpi=150)
    #         plt.close(fig1)

    #         # --- Plot 2: Monte Carlo Convergence (diagnostic) ---
    #         # Clip first fex obs to smooth plot
    #         fig2, ax2 = plt.subplots(figsize=(8, 6))

    #         for j in range(cdso_MC_hist_bps.shape[1]):
    #             ax2.plot(np.arange(2, cdso_MC_hist_bps.shape[0]),
    #                     cdso_MC_hist_bps[2:, j],
    #                     label=f"Strike {int(strike_offsets_bps[j])}bps",
    #                     alpha=0.7)

    #         ax2.set_xlabel("Number of Monte Carlo Samples")
    #         ax2.set_ylabel("Running Mean Option Price (bps)")
    #         # ax2.set_title("Monte Carlo Convergence Diagnostic")
    #         ax2.legend(fontsize=8, loc='best')
    #         ax2.grid(True)

    #         fig2.tight_layout()
    #         fig2.savefig(os.path.join(save_path, f"CDSO_MC_convergenceLHC_{firm}_m{i}.png"), dpi=150)
    #         plt.close(fig2)


    #         # Clip first fex obs to smooth plot
    #         fig3, ax3 = plt.subplots(figsize=(8, 6))

    #         for j in range(cdso_MC_hist_bps.shape[1]):
    #             ax3.plot(np.arange(2, cdso_MC_hist_bps_cir.shape[0]),
    #                     cdso_MC_hist_bps_cir[2:, j],
    #                     label=f"Strike {int(strike_offsets_bps[j])}bps",
    #                     alpha=0.7)

    #         ax3.set_xlabel("Number of Monte Carlo Samples")
    #         ax3.set_ylabel("Running Mean Option Price (bps)")
    #         # ax3.set_title("Monte Carlo Convergence Diagnostic")
    #         ax3.legend(fontsize=8, loc='best')
    #         ax3.grid(True)

    #         fig3.tight_layout()
    #         fig3.savefig(os.path.join(save_path, f"CDSO_MC_convergenceCIR_{firm}_m{i}.png"), dpi=150)
    #         plt.close(fig3)




    #         # --- Plot 1: Price vs Strike Offset ---
    #         fig1, ax1 = plt.subplots(figsize=(8, 6))

    #         ax1.plot(barriers_percentage, digital_MC, 'o-', color='navy', alpha=0.8,label='LHC Model, Simulated')
    #         ax1.plot(barriers_percentage, digital_MC_cir, 'o-', color='forestgreen', alpha=0.8,label='CIR Model, Simulated')
    #         # ax1.plot(strike_offsets_bps, cdso_MC_bps_cir_fourier, 'o-', color='grey', alpha=0.8,label='CIR Model, Fourier')

    #         ax1.set_xlabel("Percentage of spot CDS rate")
    #         ax1.set_ylabel("Digital barrier Option Price (bps)")
    #         # ax1.set_title("Digital barrier option Price")
    #         ax1.grid(True)
    #         ax1.legend(fontsize=8, loc='best')

    #         fig1.tight_layout()
    #         fig1.savefig(os.path.join(save_path, f"Digital_MC_vs_Strike_{firm}_m{i}.png"), dpi=150)
    #         plt.close(fig1)

    #         # --- Plot 2: Monte Carlo Convergence (diagnostic) ---
    #         # Clip first fex obs to smooth plot
    #         fig2, ax2 = plt.subplots(figsize=(8, 6))

    #         for j in range(digital_MC_hist.shape[1]):
    #             ax2.plot(np.arange(2, digital_MC_hist.shape[0]),
    #                     digital_MC_hist[2:, j],
    #                     label=f"Barrier {int(barriers[j])}bps",
    #                     alpha=0.7)

    #         ax2.set_xlabel("Number of Monte Carlo Samples")
    #         ax2.set_ylabel("Running Mean Option Price (bps)")
    #         # ax2.set_title("Monte Carlo Convergence Diagnostic, Digital")
    #         ax2.legend(fontsize=8, loc='best')
    #         ax2.grid(True)

    #         fig2.tight_layout()
    #         fig2.savefig(os.path.join(save_path, f"Digital_MC_convergenceLHC_{firm}_m{i}.png"), dpi=150)
    #         plt.close(fig2)


    #         # Clip first fex obs to smooth plot
    #         fig3, ax3 = plt.subplots(figsize=(8, 6))

    #         for j in range(digital_MC_hist.shape[1]):
    #             ax3.plot(np.arange(2, digital_MC_hist.shape[0]),
    #                     digital_MC_hist_cir[2:, j],
    #                     label=f"Barrier {int(barriers[j])}bps",
    #                     alpha=0.7)

    #         ax3.set_xlabel("Number of Monte Carlo Samples")
    #         ax3.set_ylabel("Running Mean Option Price (bps)")
    #         # ax3.set_title("Monte Carlo Convergence Diagnostic, Digital")
    #         ax3.legend(fontsize=8, loc='best')
    #         ax3.grid(True)

    #         fig3.tight_layout()
    #         fig3.savefig(os.path.join(save_path, f"Digital_MC_convergenceCIR_{firm}_m{i}.png"), dpi=150)
    #         plt.close(fig3)



    #         ### Lookback errors.

    #         # --- Plot 2: Monte Carlo Convergence (diagnostic) ---
    #         # Clip first fex obs to smooth plot
    #         fig2, ax2 = plt.subplots(figsize=(8, 6))
    #         x_vals = np.arange(2, look_MC_hist.shape[0])
    #         # Plot the two convergence lines
    #         ax2.plot(x_vals, look_MC_hist[2:] * 10000, alpha=0.7, color='orange', label='LHC')
    #         ax2.plot(x_vals, look_MC_hist_cir[2:] * 10000, alpha=0.7, color='green', label='AFC')
    #         # Add end price text annotations
    #         end_x = x_vals[-1]
    #         end_y_lhc = look_MC * 10000
    #         end_y_cir = look_MC_cir * 10000
    #         ax2.text(end_x, end_y_lhc, f"{end_y_lhc:.2f} bps", color='orange', fontsize=9,
    #                 va='bottom', ha='right', fontweight='bold')
    #         ax2.text(end_x, end_y_cir, f"{end_y_cir:.2f} bps", color='green', fontsize=9,
    #                 va='bottom', ha='right', fontweight='bold')

    #         # Labels and styling
    #         ax2.set_xlabel("Number of Monte Carlo Samples")
    #         ax2.set_ylabel("Running Mean Lookback Option Price (bps)")
    #         # ax2.set_title("Monte Carlo Convergence Diagnostic, Lookback")
    #         ax2.legend(fontsize=8, loc='best')
    #         ax2.grid(True)
    #         fig2.tight_layout()
    #         fig2.savefig(os.path.join(save_path, f"Lookback_MC_convergence_{firm}_m{i}.png"), dpi=150)
    #         plt.close(fig2)




    #         # Minimum distribution.
    #         fig4, ax4 = plt.subplots(figsize=(8, 6))

    #         # Plot histograms (normalized to probability densities)
    #         ax4.hist(look_cds_min_cir * 10000, bins=40, alpha=0.6, density=True,
    #                 label='AFC Model', color='tab:blue', edgecolor='black')
    #         ax4.hist(look_cds_min * 10000, bins=40, alpha=0.6, density=True,
    #                 label='LHCC Model', color='tab:orange', edgecolor='black')

    #         # Axis labels and title
    #         ax4.set_xlabel("Minimum CDS (bps)")
    #         ax4.set_ylabel("Density")
    #         # ax4.set_title("Distribution of Minimum CDS spreads")

    #         # Add legend and grid
    #         ax4.legend(fontsize=8, loc='best')
    #         ax4.grid(True, linestyle='--', alpha=0.6)

    #         # Tight layout and save
    #         fig4.tight_layout()
    #         fig4.savefig(os.path.join(save_path, f"Lookback_Distribution_Comparison_{firm}_m{i}.png"), dpi=150)
    #         plt.close(fig4)
  
    #     ### Save option output.
        
    #         filepath = os.path.join(directory, f"Option_data_{firm}_X{i}.npz")

    #     # Ensure directory exists
    #         os.makedirs(directory, exist_ok=True)

    #         np.savez(filepath,
    #             cdso_poly=cdso_actual,
    #             cdso_lhc_MC=cdso_MC,
    #             cdso_cir_MC=cdso_MC_cir,
    #             digital_lhc=digital_MC,
    #             digital_cir_MC=digital_MC_cir,                
    #             look_MC_lhc=look_MC,
    #             look_MC_cir=look_MC_cir,   
    #             look_lhc_min = look_cds_min,            
    #             look_cir_min = look_cds_min_cir,            
    #             cdso_strikes = strikes,
    #             digital_barriers = barriers,
    #             cds_obs = cds0
    #          )
            



# Redo Plots but include both Xdims
# Run again for full specification.
    from Models.LHCModels.LHC_wGamma1 import LHC_single as LHC_Gamma1
    # firms = ['DANBNK','MONTE'] # IG,IG,HY,IG
    firms = ['MONTE'] # IG,IG,HY,IG

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

        directory = f"./Results_noMPR/{firm}"
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
            lhc = LHC_Gamma1(r,delta,tenor)
            Y_dim,X_dim = 1,i
            X_now = XnLHCK[-1,:] # Want to predict forward in time.
            lhc.initialise_LHC(Y_dim,X_dim,X0=X_now)
            lhc.flatten_params()
            # Set with optimal parameters
            lhc.unflatten_params(final_paramLHCK[:2*X_dim+1])
            # Set also P params.
            gamma1 = np.array([final_paramLHCK[2*X_dim]])
            lhc_P = lhc.build_P_params(params=final_paramLHCK[2*X_dim+1:],gamma1=gamma1)
            # Then, we are ready to price in every dimension.
            # Strike grid. Base around forward spread
            # We consider this setup
            t_now= 0.0
            t0 = 1.0 # Maturity of CDSoption.
            t_mat = t0 + 5.0 # So CDS with 5 yr mat.
            cds0_lhc = lhc.CDS_model(t_obs = np.array([t_now]),T_M_grid=np.array([[t_mat-t0]]),
                                CDS_obs=None,t0=np.array([t_now]),X_in = XnLHCK[-1,:].reshape((X_dim,1)),
                                Y_in = np.array([YnLHCK[-1]]).reshape((1,1)))

            ## The inital value at en of data is just observed CDS spread.
            cds0 = CDS_obs[-1,2]

            # Then strikes around the above. Say PM 150 Bps.
            # strike_diff_grid = np.linspace(-60,60,20) / 10000
            strike_diff_grid = np.linspace(-40,120,int((120+40)/20)+1) / 10000
            strikes = (cds0 + strike_diff_grid).flatten()
            # Set Simulate option sizes.
            chi0 = np.concatenate((np.array([1]), XnLHCK[-1,:]))
            # Set discretization and number of simul. 1m simuls, fairly coarse grid
            N, M = 10000, 100
            # N, M = 400,100
            seed = 2000
            # Take Legendre poly at 20. 
            leg_deg = 20 # 20 . This is were it explodes in X_dim=3

            ### Get CDSO using theoretical values.
            # Note, we need to discound too, see top vof p. 19
            # To check if actually worth running many. 
            # cdso_actual = lhc.get_cdso_price(t_now,t0,t_mat,chi0[0],chi0[1:],np.array([0.005]),n_max=9)
            # cdso_MC_hist,cdso_MC = lhc.get_cdso_price_MC(t_now,t0,t_mat,np.array([0.005]),chi0,N,M,seed=1000) 
            # print(f'Test: Actual CDSO price {cdso_actual}')
            # print(f'Test: MC CDSO price {cdso_MC}')
            # Must assume it to be correct. 
            cdso_actual_g = lhc.get_cdso_price(t_now,t0,t_mat,chi0[0],chi0[1:],strikes,n_max=leg_deg)
            cdso_MC_hist_g,cdso_MC_g = lhc.get_cdso_price_MC(t_now,t0,t_mat,strikes,chi0,N,M,seed=seed)
            # N, M = 100,500

            ######### DIGITAL OPTION PRICING. ##############
            # Define Barriers as some percentage of the CDS at current timepoint.
            barriers_percentage = np.array([(1+i*0.05) for i in range(1,11)])
            barriers = cds0*barriers_percentage
            T_mat_barrier = t_now + (t0-t_now)/2 # Matures halfway to the CDS
            digital_MC_hist_g,digital_MC_g = lhc.get_digital_barrier_price_MC(t_now,t0,t_mat,T_mat_barrier,barriers,
                                                                        chi0,N,M,seed=seed)


            ######### Lookback OPTION PRICING. ##############
            # Consider CDS at expiry.
            look_MC_hist_g,look_MC_g,look_cds_min_g =  lhc.get_lookback_price_MC(t_now,t0,t_mat,T_mat_barrier,
                                                                        chi0,N,M,seed=seed)



            ## Plotting. Use functionality.
            # --- convert to basis points ---
        
        
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

            save_path = f"./Results_noMPR/Options_full"

        #     save_path = directory + f"/Options"
            
            os.makedirs(save_path, exist_ok=True)

            # --- Plot 1: Price vs Strike Offset ---
            fig1, ax1 = plt.subplots(figsize=(8, 6))

            ax1.plot(strike_offsets_bps, cdso_MC_bps, 'o-', color='navy', alpha=0.8,label='LHC Model, Simulated')
            ax1.plot(strike_offsets_bps, cdso_actual*10000, 'o-', color='red', alpha=0.8,label=f'LHC Model, Legendre n={leg_deg}')
            ax1.plot(strike_offsets_bps, cdso_MC_g*10000, 'x-', color='magenta', alpha=0.8,label='LHC Model full, Simulated')
            ax1.plot(strike_offsets_bps, cdso_actual_g*10000, 'o-', color='magenta', alpha=0.8,label=f'LHC Model full, Legendre n={leg_deg}')
            ax1.plot(strike_offsets_bps, cdso_MC_bps_cir, 'o-', color='forestgreen', alpha=0.8,label='CIR Model, Simulated')
            # ax1.plot(strike_offsets_bps, cdso_MC_bps_cir_fourier, 'o-', color='grey', alpha=0.8,label='CIR Model, Fourier')

            ax1.set_xlabel("Strike Offset (bps)")
            ax1.set_ylabel("CDS Option Price (bps)")
            # ax1.set_title("CDS Option Price vs Strike Offset")
            ax1.grid(True)
            ax1.legend(fontsize=8, loc='best')

            fig1.tight_layout()
            fig1.savefig(os.path.join(save_path, f"CDSO_MC_vs_Strike_{firm}_m{i}.png"), dpi=150)
            plt.close(fig1)


            # --- Plot 1: Price vs Strike Offset ---
            fig1, ax1 = plt.subplots(figsize=(8, 6))

            ax1.plot(barriers_percentage, digital_MC, 'o-', color='navy', alpha=0.8,label='LHC Model, Simulated')
            ax1.plot(barriers_percentage, digital_MC_g, 'o-', color='magenta', alpha=0.8,label='LHC Model full, Simulated')
            ax1.plot(barriers_percentage, digital_MC_cir, 'o-', color='forestgreen', alpha=0.8,label='CIR Model, Simulated')
            # ax1.plot(strike_offsets_bps, cdso_MC_bps_cir_fourier, 'o-', color='grey', alpha=0.8,label='CIR Model, Fourier')

            ax1.set_xlabel("Percentage of spot CDS rate")
            ax1.set_ylabel("Digital barrier Option Price (bps)")
            # ax1.set_title("Digital barrier option Price")
            ax1.grid(True)
            ax1.legend(fontsize=8, loc='best')

            fig1.tight_layout()
            fig1.savefig(os.path.join(save_path, f"Digital_MC_vs_Strike_{firm}_m{i}.png"), dpi=150)
            plt.close(fig1)

            ### Lookback errors.

            # --- Plot 2: Monte Carlo Convergence (diagnostic) ---
            # Clip first fex obs to smooth plot
            # fig2, ax2 = plt.subplots(figsize=(8, 6))
            # x_vals = np.arange(2, look_MC_hist.shape[0])
            # # Plot the two convergence lines
            # ax2.plot(x_vals, look_MC_hist[2:] * 10000, alpha=0.7, color='navy', label='LHC')
            # ax2.plot(x_vals, look_MC_hist_g[2:] * 10000, alpha=0.7, color='magenta', label='LHC full')
            # ax2.plot(x_vals, look_MC_hist_cir[2:] * 10000, alpha=0.7, color='forestgreen', label='AFC')
            # # Add end price text annotations
            # end_x = x_vals[-1]
            # end_y_lhc = look_MC * 10000
            # end_y_lhc_g = look_MC_g * 10000
            # end_y_cir = look_MC_cir * 10000
            # ax2.text(end_x, end_y_lhc, f"{end_y_lhc:.2f} bps", color='navy', fontsize=9,
            #         va='bottom', ha='right', fontweight='bold')
            # ax2.text(end_x, end_y_lhc, f"{end_y_lhc_g:.2f} bps", color='magenta', fontsize=9,
            #         va='bottom', ha='right', fontweight='bold')
            # ax2.text(end_x, end_y_cir, f"{end_y_cir:.2f} bps", color='forestgreen', fontsize=9,
            #         va='bottom', ha='right', fontweight='bold')

            # # Labels and styling
            # ax2.set_xlabel("Number of Monte Carlo Samples")
            # ax2.set_ylabel("Running Mean Lookback Option Price (bps)")
            # ax2.set_title("Monte Carlo Convergence Diagnostic, Lookback")
            # ax2.legend(fontsize=8, loc='best')
            # ax2.grid(True)
            # fig2.tight_layout()
            # fig2.savefig(os.path.join(save_path, f"Lookback_MC_convergence_{firm}_m{i}.png"), dpi=150)
            # plt.close(fig2)


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
            ax2.legend(fontsize=8, loc='best')
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
            ax4.legend(fontsize=8, loc='best')
            ax4.grid(True, linestyle='--', alpha=0.6)

            # Tight layout and save
            fig4.tight_layout()
            fig4.savefig(os.path.join(save_path, f"Lookback_Distribution_Comparison_{firm}_m{i}.png"), dpi=150)
            plt.close(fig4)

            # Save add on data.
            filepath = os.path.join(directory, f"Option_data_{firm}_X{i}_full.npz")

            # Ensure directory exists
            os.makedirs(directory, exist_ok=True)

            np.savez(filepath,
                cdso_lhc_MC = cdso_MC_g,
                digital_lhc = digital_MC_g,
                look_MC_lhc = look_MC_g,
                look_lhc_min = look_cds_min_g,            
                cdso_strikes = strikes,
                digital_barriers = barriers,
             )
            

        ### Save option output.
        