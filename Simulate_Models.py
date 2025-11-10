if __name__ == "__main__":
    from Models.BaselineCIR_alternative.CIR_Multifactor import CIRIntensity as CIR
    from Models.BaselineCIR_alternative.Gamma_solver import DeterministicGamma as Gamma_class 
    from Models.LHCModels.LHC_single import LHC_single as LHC
    from Models.LHCModels.LHC_single import get_CDS_Model,kalmanfilter_opt, rebuild_lhc_struct, cds_value, solve_mu1, compute_stationary
    # from Models.LHCModels.LHC_single_XY import LHC_single as LHC
    # from Models.LHCModels.LHC_single_XY import get_CDS_Model, rebuild_lhc_struct, cds_value, solve_mu1, compute_stationary
    import pandas as pd


    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from scipy.stats import norm 

    ## Some global parameters (simulate forward in time, grid fineness.)
    # Also some realistic size (TODO)
    # Simulat eover longer time horizon
    T,M = 10, 300
    # Loop over the two different X_dim comps.
    # for X_dim in [1,2,3]: #,2,3]:

    #     ### Simulate 1 LHC dataset with specific parameter Choises.
    #     lhc = LHC(0.0252,0.4,0.25)
    #     Y_dim,m = 1,X_dim
    #     # Here, parameters are set already
    #     rng = np.random.default_rng(245)
    #     X0 = 0.3
    #     chi0 = np.array([1] + [X0]*m)
    #     lhc.initialise_LHC(Y_dim,m,X0=X0,rng=rng)
    #     lhc.flatten_params()
    #     params = lhc.flatten_params()
    #     lhc.unflatten_params(params[:2*m+1])
    #      ### Just for fairness, initialise at mu1 for z
    #     # Set also P params. 
    #     lhc_P = lhc.build_P_params(rng=rng)
    #     # Start value stationary ones. 
    #     mu1 = lhc.solve_mu1(lhc.kappa,lhc.theta,lhc.gamma1, lambda_i=lhc.lambda_i)
    #     mu = lhc.compute_stationary(lhc.kappa,lhc.theta,lhc.m,lhc.gamma1,mu1,lhc.lambda_i)
    #     chi0 = np.append([1] , mu)
    #     params_actual = [lhc.kappa, lhc.theta,lhc.gamma1,lhc.lambda_i,lhc.sigma, lhc.sigma_err]
    #     # params_actual = np.array([0.546,0.421,
    #     #                           0.624,0.512,
    #     #                           0.205,
    #     #                           0,0,
    #     #                           0.5,0.3,
    #     #                           0.001])
    #     # lhc.unflatten_params(params_actual[:2*m+1])
    #     # _= lhc.build_P_params(params_actual[2*m+1:],lhc.gamma1)
    #     # Simulate. We are using an Euler discretization. 
    #     # Start at 0.5 also for aloowing for more jump op and down. Again, likly too large initial cov
    #     # Use same seed to reproduce same randomness.
    #     mat_grid = np.array([1,3,5,7,10]) # Typical maturity grid
    #     # mat_grid = np.array([5]) 

    #     n_mat = mat_grid.shape[0]
    #     T_path, chi_Q = lhc.simul_latent_states(chi0=chi0,T=T,M=M,n_mat=n_mat,seed=200,
    #                                             scheme='Milstein',measure='P')
    #     # Try to simulate Z instead and see if it makes it better...
    #     T_path, Z_Q = lhc.simul_Z(chi0=chi0[1:],T=T,M=M,n_mat=n_mat,seed=200,
    #                                             scheme='Euler',measure='P')
    #     Z_ones = np.hstack([np.ones((Z_Q.shape[0],1)),Z_Q])
    #     # Holld maturity to be 5
    #     t_mat_grid = np.ascontiguousarray(mat_grid[:, None] + T_path[None, :])   # shape (len(T_M_grid), len(t_obs))

    #     t0 = T_path

    #     kappa, theta, gamma1 = lhc.kappa,lhc.theta,lhc.gamma1[0]
    #     r = lhc.r
    #     Y_dim = lhc.Y_dim
    #     delta = lhc.delta
    #     tenor = lhc.tenor 
    #     lhc_numba = rebuild_lhc_struct(kappa, theta, gamma1, r, Y_dim, delta, tenor)


    #     # Draw noise vector:
    #     save_path = "./Simulation_studies/"   # <--- change to your path
    #     color_cycle = plt.cm.tab10.colors  

    #     R = norm.rvs(size = (t_mat_grid.shape[0]*t_mat_grid.shape[1]),scale = lhc.sigma_err).reshape(t_mat_grid.shape) # simulate at beginning - faster!
    #     CDS_simul_actual = get_CDS_Model(T_path, t0, t_mat_grid, chi_Q.T, lhc_numba)  
    #     CDS_simul = CDS_simul_actual+ R
    #     # Test of value:
    #     print(cds_value(lhc_numba, t0[0], t0[0], t_mat_grid[:,0],CDS_simul[:,0]) @ chi_Q[0,:])
    #     print(np.min(CDS_simul),np.max(CDS_simul))
    #     CDS_simul_Z_actual = get_CDS_Model(T_path, t0, t_mat_grid, Z_ones.T, lhc_numba)  
    #     # Reestimation of the Process. Try Kalman filter on the recreated CDS spreads.
    #     # Kalman automatically initiates at random. 
    #     lhc_kalman_params,  Xn,Zn, Pn, se= lhc.run_n_kalmans(T_path, t_mat_grid, CDS_simul.T,base_seed=2000,n_restarts=1)
    #     # Xn_kalman,Yn_kalman= Xn[:,1:], Xn[:,0]
    #     Xn_kalman,Yn_kalman = lhc.kalman_X_Y(T_path,Xn)
    #     # Recalculate CDS spreads, if value approach:
    #     # Get CDS Spreads from kalman
    #     CDS_kalman = lhc.CDS_model(T_path,  t_mat_grid, CDS_simul.T , 
    #                                t0=T_path, X_in=Xn_kalman.T,Y_in=Yn_kalman) 


    #     np.set_printoptions(precision=4, suppress=True)  # fewer decimals, no scientific notation
    #     print(f'Optimal parameters, Kalman: {lhc_kalman_params}')
    #     # print(f'Optimal Parameters Filipovic {lhc_params}')
    #     print(f'Actual Parameters: {params_actual}')
    #     param_lhc_names = [f'kappa{i}' for i in range(1,X_dim+1)]+ [f'theta{i}' for i in range(1,X_dim+1)]
    #     param_lhc_names += ['gamma1'] + [f'lambda{i}' for i in range(1,X_dim+1)]
    #     param_lhc_names +=  [f'sigma{i}' for i in range(1,X_dim+1)]
    #     param_lhc_names += [f'sigma_err']

    #     df = pd.DataFrame({
    #         'Parameter': param_lhc_names,
    #         'Estimated Kalman': lhc_kalman_params,
    #         # 'Estimated Filipovic': np.append(lhc_params,np.zeros(lhc_kalman_params.shape[0] - lhc_params.shape[0])),
    #         'True': np.concatenate(params_actual),
    #     })
    #     df['Abs Error, Kalman'] = np.abs(df['Estimated Kalman'] - df['True'])
    #     df['Rel Error (%), Kalman'] = 100 * np.abs(df['Estimated Kalman'] - df['True']) / df['True']
    #     # df['Abs Error, Filipovic'] = np.abs(df['Estimated Filipovic'] - df['True'])
    #     # df['Rel Error (%), Filipovic'] = 100 * np.abs(df['Estimated Filipovic'] - df['True']) / df['True']


    #     print(df)
    #     df.to_excel(os.path.join(save_path, f'lhc_parameter_comparison_Xdim{X_dim}.xlsx'), index=False)

    #     # --- Latent states plots ---
    #     n_states = chi_Q.shape[1]  # total number of latent states

    #     fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    #     # ---------- Plot 1: Survival process (Y) ----------
    #     ax = axes[0]
    #     ax.plot(T_path, chi_Q[:, 0], "-", alpha=0.8, label="Y / Survival Process (Q, sim)", color="blue")
    #     ax.plot(T_path, Yn_kalman, "-.", alpha=0.9, label="Y (Kalman)", color="green")
    #     # ax.plot(T_path, Y, "--", alpha=0.9, label="Y (Filipovic)", color="gold")

    #     ax.set_ylabel("Y / Survival Process")
    #     ax.set_title("Latent States: Y and X Dynamics")
    #     ax.legend(loc="best")
    #     ax.grid(True, linestyle="--", alpha=0.6)

    #     # ---------- Plot 2: X states ----------
    #     ax = axes[1]
    #     for i in range(1, n_states):
    #         state_name = f"X{i}"
    #         ax.plot(T_path, chi_Q[:, i], "-", alpha=0.8, label=f"{state_name} (Q, sim)", color=color_cycle[(i-1) % len(color_cycle)])
    #         # ax.plot(T_path, X[i-1, :], "-.", alpha=0.9, label=f"{state_name} (Filipovic)", color=color_cycle[(i-1) % len(color_cycle)])
    #         ax.plot(T_path, Xn_kalman[:, i-1], "--", alpha=0.9, label=f"{state_name} (Kalman)", color=color_cycle[(i-1) % len(color_cycle)])

    #     ax.set_xlabel("Time (years)")
    #     ax.set_ylabel("Latent Factors X")
    #     ax.legend(loc="best", ncol=2)
    #     ax.grid(True, linestyle="--", alpha=0.6)

    #     fig.tight_layout()
    #     fig.savefig(os.path.join(save_path, f"SimulLHC_LatentStates_Combined_Xdim{X_dim}.png"), dpi=150)
    #     plt.close(fig)


    #     # --- CDS spreads plot ---
    #     fig, ax = plt.subplots(figsize=(10,5))

    #     for i in range(CDS_simul.shape[0]):
    #         ax.plot(T_path, CDS_simul_actual[i,:], "-", alpha=0.9, label=f"CDS, T_mat={mat_grid[i]}", color='red')
    #         ax.plot(T_path, CDS_simul[i,:], "-", alpha=0.7, label=f"CDS corrupted, T_mat={mat_grid[i]}", color='black')
    #         ax.plot(T_path, CDS_kalman[:, i], "--", alpha=0.9, label=f"CDS Kalman, T_mat={mat_grid[i]}", color='green')
    #         # ax.plot(T_path, CDS_model[:, i], "--", alpha=0.9, label=f"CDS Filipovic, T_mat={mat_grid[i]}", color='yellow')

    #     ax.set_xlabel("Time (years)")
    #     ax.set_ylabel("CDS Spreads")
    #     ax.set_title("CDS Spreads Comparison")
    #     ax.legend()
    #     fig.tight_layout()
    #     fig.savefig(os.path.join(save_path, f"SimulLHC_CDS_Spreads_Xdim{X_dim}.png"), dpi=150)
    #     plt.close(fig)



    T,M = 10, 200
    # Loop over the two different X_dim comps.
    for X_dim in [1,2,3]:

        #### CIR SIMULATION. ######
        save_path = "./Simulation_studies/"   # <--- change to your path

        seed = 145
        cir = CIR(0.0252, 0.4, 0.25,X_dim)
        r = cir.r
        delta = cir.delta
        tenor = cir.tenor 
        # params_actual =np.array([-0.1115+0.2247,0.0611,0.0702,0.2247,0.0611,0.003]) # Use params from a article
        cir.set_params(params=None, seed=seed)
        print(cir.kappa,cir.theta,cir.sigma,cir.lambda1,cir.sigma_err)
        params_cir = np.concatenate([cir.kappa,cir.theta,cir.sigma,cir.lambda1,cir.sigma_err])
        # Simulate. We are using an Euler discretization. 
        # Set initial lambda to the one we would get in a LHC model.
        # initial values - all under Q here...
        alpha = 2 * cir.kappa*  cir.theta /  cir.sigma**2
        beta = 2 * cir.kappa /  cir.sigma**2
        # CIR values.
        lambda0 = alpha / beta # start below long tern
        # This is under Q 
        # T_return,lambda_eul_Q = cir.simulate_intensity(lambda0=lambda0,T=T,M=M,scheme="Euler")
        T_return,lambda_mil_Q = cir.simulate_intensity(lambda0=lambda0,T=T,M=M,
                                                       scheme="Milstein",seed=seed,measure='P')

        mat_grid = np.array([1,3,5,7,10])
        # mat_grid = np.array([1,5,10])
        t_mat_grid = np.ascontiguousarray(mat_grid[:, None] + T_return[None, :])   # shape (len(T_M_grid), len(t_obs))

        # Get true CDS spreads. 
        CDS_cir = np.ones((t_mat_grid.T.shape))
        for i in range(t_mat_grid.shape[1]):
            lambda_curr  = np.array([lambda_mil_Q[i]])
            mat_curr = t_mat_grid[:,i]
            CDS_cir[i,:] = cir.cds_spread(lambda_curr,params_cir,T_return[i],mat_curr)
            # CDS_cir_test = cir.cds_spread(lambda_curr,params_cir,0.0,mat_grid)

        # Add the citter to spreads.

        ### See how spreads, and states look

        color_cycle = plt.cm.tab10.colors  

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6), sharey=False)

        # === LATENT STATES (all X[:,i]) ===
        n_states = lambda_mil_Q.shape[1]  # number of latent factors

        for i in range(n_states):
            color = color_cycle[i % len(color_cycle)]

            # Optionally add simulated versions if available (like lambda_mil_Q / lambda_mil_P)
            if 'lambda_mil_Q' in locals():
                ax1.plot(T_return, lambda_mil_Q[:,i], "-", alpha=0.7, label=f"Simulated X{i+1} (CIR)", color=color)

        ax1.set_xlabel("Time (years)")
        ax1.set_ylabel("Latent State")
        ax1.legend()
        ax1.set_title("Latent States")

        # === CDS spreads (simulated vs Kalman estimates) ===
        for i in range(CDS_cir.shape[1]):
            color = color_cycle[i % len(color_cycle)]
            ax2.plot(T_return, CDS_cir[:, i], "-", alpha=0.7,
                    label=f"CDS Sim, T_mat={mat_grid[i]}", color=color)


        ax2.set_xlabel("Time (years)")
        ax2.set_ylabel("CDS Spreads")
        ax2.legend()
        ax2.set_title("Spreads")

        fig.tight_layout()

        # Save figure
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, f"True_Simul_CIR_Xdim{X_dim}.png"), dpi=150)
        plt.close(fig)


        ### See if possible to estimate the parameters. For that, need to turn into integrated lambda.
        # Need to generate the deterministic lambdas.
        # We dont need to callibrate deterministic lambda. We can simply take log of the solution to Ricatti eqs.
        model = Gamma_class(r, delta, tenor)
        t_mats = np.concatenate(([0],mat_grid))
        extrapolate_grid = np.array([i  for i in range(int(np.max(mat_grid))+ 1)])    

        Gamma = np.zeros((CDS_cir.shape[0], extrapolate_grid.shape[0]))
        cali_params = np.zeros((CDS_cir.shape[0], mat_grid.shape[0]))
        survival = np.zeros((CDS_cir.shape[0], extrapolate_grid.shape[0]))

        for t_idx in range(CDS_cir.shape[0]):
            # Calibrate back hazard rates
            t_grid_payments = np.array([tenor*i for i in range(int(np.max(mat_grid)/tenor)+1)])   
            cali_params[t_idx, : ] = model.calibrate_deterministic(CDS_cir[t_idx,:] , mat_grid, 0.0, t_grid_payments)
            # Generate the survial probabilities/survival process
            for i in range(extrapolate_grid.shape[0]):
                Gamma[t_idx,i] = model.Gamma_fun(cali_params[t_idx, : ],extrapolate_grid[i],t_mats)
                survival[t_idx,i] = np.exp(-Gamma[t_idx,i] )


        Gamma_kalman = Gamma[:,np.isin(extrapolate_grid, mat_grid).flatten()]

        # Kalman noise.
        R = norm.rvs(size = (Gamma_kalman.shape[0]*Gamma_kalman.shape[1]),scale = cir.sigma_err).reshape(Gamma_kalman.shape) # simulate at beginning - faster!
       
        Gamma_kalman_noise = Gamma_kalman + R

        ## Try CIR++ Estimation procedure


        # Select only params at maturity. 
        params_cir_est, Xn_cir,Zn_cir, Pn_cir,se_cir = cir.run_kalman_filter(T_return,t_mat_grid,Gamma_kalman,seed=2000)
        
        ### illustrate fitted process.
        
        color_cycle = plt.cm.tab10.colors  

        fig, (ax1) = plt.subplots(1,1, figsize=(14,6), sharey=False)

        for i in range(Gamma_kalman.shape[1]):
            color = color_cycle[i % len(color_cycle)]
            ax1.plot(T_return, Zn_cir[:, i], "--", alpha=0.9, color=color, label=f"Kalman Gamma{mat_grid[0]}")
            ax1.plot(T_return, Gamma_kalman[:, i], "--", alpha=0.9, color=color, label=f"Actual Gamma{mat_grid[0]}")

        ax1.set_xlabel("Time (years)")
        ax1.set_ylabel("Integrated Gamma")
        ax1.legend()
        ax1.set_title("Integrated Gamma")

        fig.tight_layout()

        # Save figure
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, f"Gamma_fits.png"), dpi=150)
        plt.close(fig)
        
        # # Then get CDS spread
        CDS_cir_model = np.ones((t_mat_grid.T.shape))
        for i in range(t_mat_grid.shape[1]):
            lambda_curr  = np.array([Xn_cir[i,:]])
            mat_curr = t_mat_grid[:,i]
            CDS_cir_model[i,:] = cir.cds_spread(lambda_curr,params_cir_est,T_return[i],mat_curr)
            print(f'Done with {(i+1)/t_mat_grid.shape[1]} %')

        np.set_printoptions(precision=4, suppress=True)

        # Param names: 
        param_cir_names = [f'kappa{i}' for i in range(1,X_dim+1)]+ [f'theta{i}' for i in range(1,X_dim+1)]
        param_cir_names += [f'sigma{i}' for i in range(1,X_dim+1)]
        param_cir_names += [f'lambda{i}' for i in range(1,X_dim+1)]+ [f'sigma_err']
        
        df = pd.DataFrame({
            'Parameter': param_cir_names,
            'Estimated': params_cir_est,
            'True': params_cir,
        })
        df['Abs Error'] = np.abs(df['Estimated'] - df['True'])
        df['Rel Error (%)'] = 100 * np.abs(df['Estimated'] - df['True']) / df['True']

        print(df)
        df.to_excel(os.path.join(save_path, f'cir_parameter_comparison_Xdim{X_dim}.xlsx'), index=False)


        # --- Latent state (simulated vs Kalman estimate) ---

        color_cycle = plt.cm.tab10.colors  

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6), sharey=False)

        # === LATENT STATES (all X[:,i]) ===
        n_states = Xn_cir.shape[1]  # number of latent factors

        for i in range(n_states):
            color = color_cycle[i % len(color_cycle)]
            ax1.plot(T_return, Xn_cir[:, i], "--", alpha=0.9, color=color, label=f"Kalman X{i+1}")

        # Optionally add simulated versions if available (like lambda_mil_Q / lambda_mil_P)
            if 'lambda_mil_Q' in locals():
                ax1.plot(T_return, lambda_mil_Q[:,i], "-", alpha=0.7, label=f"Simulated X{i+1} (CIR)", color=color)

        ax1.set_xlabel("Time (years)")
        ax1.set_ylabel("Latent State")
        ax1.legend()
        ax1.set_title("Latent States")

        # === CDS spreads (simulated vs Kalman estimates) ===
        for i in range(CDS_cir.shape[1]):
            color = color_cycle[i % len(color_cycle)]
            ax2.plot(T_return, CDS_cir[:, i], "-", alpha=0.7,
                    label=f"CDS Sim, T_mat={mat_grid[i]}", color=color)
            ax2.plot(T_return, CDS_cir_model[:, i], "--", alpha=0.9,
                    label=f"CDS Kalman, T_mat={mat_grid[i]}", color=color)

        ax2.set_xlabel("Time (years)")
        ax2.set_ylabel("CDS Spreads")
        ax2.legend()
        ax2.set_title("Spreads")

        fig.tight_layout()

        # Save figure
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, f"SimulCIR_states_vs_Kalman_Xdim{X_dim}.png"), dpi=150)
        plt.close(fig)

        print(f'Done with CIR({X_dim})')

    # SEction to investigate misspecification.




### tesitn parameters:
        ### 0: Test if true values actually somewhat reproduce true traject and paths.
        ### Simulate 1 LHC dataset with specific parameter Choises.
        # lhc = LHC(0.025,0.4,0.25)
        # Y_dim,m = 1,X_dim
        # # Here, parameters are set already
        # rng = np.random.default_rng(1000)
        # X0 = 0.3
        # chi0 = np.array([1] + [X0]*m)
        # lhc.initialise_LHC(Y_dim,m,X0=X0,rng=rng)
        # lhc.flatten_params()
        # params = lhc.flatten_params()
        # lhc.unflatten_params(params[:2*m+1])
        #  ### Just for fairness, initialise at mu1 for z
        # mu1 = lhc.solve_mu1(lhc.kappa,lhc.theta,lhc.gamma1)
        # mu = lhc.compute_stationary(lhc.kappa,lhc.theta,lhc.m,lhc.gamma1,mu1)
        # chi0 = np.append([1] , mu)
        # # Set also P params. 
        # lhc_P = lhc.build_P_params(rng=rng)
        # params_actual = [lhc.kappa, lhc.theta,lhc.gamma1,lhc.lambda_i,lhc.sigma, lhc.sigma_err]

        # # Simulate. We are using an Euler discretization. 
        # # Start at 0.5 also for aloowing for more jump op and down. Again, likly too large initial cov
        # # Use same seed to reproduce same randomness.
        # mat_grid = np.array([1,3,5,7,10]) # Typical maturity grid
        # # mat_grid = np.array([5]) 

        # n_mat = mat_grid.shape[0]
        # T_path, chi_Q = lhc.simul_latent_states(chi0=chi0,T=T,M=M,n_mat=n_mat,seed=200,scheme='Milstein')


        # # Holld maturity to be 5
        # t_mat_grid = np.ascontiguousarray(mat_grid[:, None] + T_path[None, :])   # shape (len(T_M_grid), len(t_obs))

        # t0 = T_path

        # kappa, theta, gamma1 = lhc.kappa,lhc.theta,lhc.gamma1[0]
        # r = lhc.r
        # Y_dim = lhc.Y_dim
        # delta = lhc.delta
        # tenor = lhc.tenor 
        # lhc_numba = rebuild_lhc_struct(kappa, theta, gamma1, r, Y_dim, delta, tenor)


        # # Draw noise vector:
        # save_path = "./Simulation_studies/"   # <--- change to your path
        # color_cycle = plt.cm.tab10.colors  

        # R = norm.rvs(size = (t_mat_grid.shape[0]*t_mat_grid.shape[1]),scale = lhc.sigma_err).reshape(t_mat_grid.shape) # simulate at beginning - faster!
        # CDS_simul_actual = get_CDS_Model(T_path, t0, t_mat_grid, chi_Q.T, lhc_numba)  
        # CDS_simul = CDS_simul_actual+ R
        
        # ## Test 
        # params_p = np.array(params_actual)[2*m+1:].flatten()
        # params_q =  np.array(params_actual)[:2*m+1].flatten()
        # lambda_i = params_p[:m]
        # kappa, theta, gamma1 = params_q[:m],params_q[m:2*m], params_q[-1]
        # kappa_p = kappa - lambda_i
        # theta_p = (kappa*theta) / kappa_p
        # theta_p[-1] = theta_p[-1] + lambda_i[-1] / kappa_p[-1]    
        # lhc_p = rebuild_lhc_struct(kappa_p, theta_p, gamma1, r, Y_dim, delta, tenor)
        # lhc_q = rebuild_lhc_struct(kappa, theta, gamma1, r, Y_dim, delta, tenor)
        # # Reestimation of the Process. Try Kalman filter on the recreated CDS spreads.
        # # Kalman automatically initiates at random. 
        # neg_loglik,Zn_kalman,CDS_kalman,Pn  = kalmanfilter_opt(np.array(params_actual).flatten(), T_path,T_path,t_mat_grid, CDS_simul.T,lhc_p,lhc_q,X0)
        # Xn_kalman,Yn_kalman = lhc.kalman_X_Y(T_path,Zn_kalman)
        # n_states = chi_Q.shape[1]  # total number of latent states

        # fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # # ---------- Plot 1: Survival process (Y) ----------
        # ax = axes[0]
        # ax.plot(T_path, chi_Q[:,0], "-", alpha=0.8, label="Y / Survival Process (Q, sim)", color="blue")
        # ax.plot(T_path, Yn_kalman, "-.", alpha=0.9, label="Y (Kalman)", color="green")
        # ax.set_ylabel("Y / Survival Process")
        # ax.set_title("Latent States: Y and X Dynamics")
        # ax.legend(loc="best")
        # ax.grid(True, linestyle="--", alpha=0.6)

        # # ---------- Plot 2: X states ----------
        # ax = axes[1]
        # for i in range(1, n_states):
        #     state_name = f"X{i}"
        #     ax.plot(T_path, chi_Q[:, i], "-", alpha=0.8, label=f"{state_name} (Simulated)", color=color_cycle[(i-1) % len(color_cycle)])
        #     ax.plot(T_path, Xn_kalman[:, i-1], "--", alpha=0.9, label=f"{state_name} (Kalman)", color=color_cycle[(i-1) % len(color_cycle)])

        # ax.set_xlabel("Time (years)")
        # ax.set_ylabel("Latent Factors X")
        # ax.legend(loc="best", ncol=2)
        # ax.grid(True, linestyle="--", alpha=0.6)

        # fig.tight_layout()
        # fig.savefig(os.path.join(save_path, f"Filter_test_Xdim{X_dim}.png"), dpi=150)
        # plt.close(fig)


        # # --- CDS spreads plot ---
        # fig, ax = plt.subplots(figsize=(10,5))

        # for i in range(CDS_simul.shape[0]):
        #     ax.plot(T_path, CDS_simul_actual[i,:], "-", alpha=0.9, label=f"CDS, T_mat={mat_grid[i]}", color='red')
        #     ax.plot(T_path, CDS_simul[i,:], "-", alpha=0.7, label=f"CDS corrupted, T_mat={mat_grid[i]}", color='black')
        #     ax.plot(T_path, CDS_kalman[:, i], "--", alpha=0.9, label=f"CDS Kalman, T_mat={mat_grid[i]}", color='green')
        #     # ax.plot(T_path, CDS_model[:, i], "--", alpha=0.9, label=f"CDS Filipovic, T_mat={mat_grid[i]}", color='yellow')

        # ax.set_xlabel("Time (years)")
        # ax.set_ylabel("CDS Spreads")
        # ax.set_title("CDS Spreads Comparison")
        # ax.legend()
        # fig.tight_layout()
        # fig.savefig(os.path.join(save_path, f"Filter_test_CDS_Spreads_Xdim{X_dim}.png"), dpi=150)
        # plt.close(fig)



    test = 1