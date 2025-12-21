if __name__ == "__main__":
    from Models.BaselineCIR_alternative.CIR_Multifactor import CIRIntensity as CIR
    from Models.BaselineCIR_alternative.Gamma_solver import DeterministicGamma as Gamma_class 
    from Models.LHCModels.LHC_single import LHC_single as LHC
    from Models.LHCModels.LHC_single import get_CDS_Model,kalmanfilter_opt, calc_gamma1,rebuild_lhc_struct, cds_value, solve_mu1, compute_stationary
    # from Models.LHCModels.LHC_single_XY import LHC_single as LHC
    # from Models.LHCModels.LHC_single_XY import get_CDS_Model, rebuild_lhc_struct, cds_value, solve_mu1, compute_stationary
    import pandas as pd


    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from scipy.stats import norm 

    # Some global parameters (simulate forward in time, grid fineness.)
    # Also some realistic size (TODO)
    # Simulat eover longer time horizon
    T,M = 10, 1000
    # Loop over the two different X_dim comps.
    for X_dim in [1,2,3]: #,2,3]:

        ### Simulate 1 LHC dataset with specific parameter Choises.
        lhc = LHC(0.00248,0.4,0.25)
        Y_dim,m = 1,X_dim
        # Here, parameters are set already
        seed = 4000
        rng = np.random.default_rng(seed)
        X0 = 0.3
        chi0 = np.array([1] + [X0]*m)
        lhc.initialise_LHC(Y_dim,m,X0=X0,rng=rng)

        # Test how drift polynomial looks. 
        # stationary_lambda = 0.05 / 100
        # if X_dim > 1:
        #     mu = lhc.compute_stationary(lhc.kappa, lhc.theta, 
        #                                         X_dim, gamma1=1, mu1=stationary_lambda, lambda_i=None)
        #     mu = mu[1]
        # else:
        #     mu = 1
        # drift = lambda l: l**2 - lhc.kappa[0]*l + lhc.kappa[0] * lhc.theta[0]*lhc.gamma1 * mu
        # gamma_range = np.array([i / 100 * lhc.gamma1*2 for i in range(0,100+1)])
        # drift_arr = np.array([drift(i) for i in gamma_range])
        # root1 = lhc.kappa[0] / 2 - np.sqrt(lhc.kappa[0]*(lhc.kappa[0]-4*lhc.gamma1*lhc.theta[0]*mu))/2
        # root2 =  lhc.kappa[0] / 2 + np.sqrt(lhc.kappa[0]*(lhc.kappa[0]-4*lhc.gamma1*lhc.theta[0]*mu))/2
        # plt.plot(gamma_range,drift_arr)
        # plt.vlines(x=lhc.gamma1, ymin = np.min(drift_arr), 
        #            ymax = np.max(drift_arr),color='red', label='gamma1')
        # plt.vlines(x=root1, ymin = np.min(drift_arr), 
        #            ymax = np.max(drift_arr),color='black',label='Lower root')
        # plt.vlines(x=root2, ymin = np.min(drift_arr), 
        #            ymax = np.max(drift_arr),color='green',label='upper root')
        # plt.legend()
        # plt.show()

        lhc.flatten_params()
        params = lhc.flatten_params()
        lhc.unflatten_params(params[:2*m+1])
         ### Just for fairness, initialise at mu1 for z
        # Set also P params. 
        lhc_P = lhc.build_P_params(rng=rng)
        # Start value stationary ones. 
        mu1 = lhc.solve_mu1(lhc.kappa_p,lhc.theta_p,lhc.gamma1, lambda_i=None)
        mu = lhc.compute_stationary(lhc.kappa_p,lhc.theta_p,lhc.m,lhc.gamma1,mu1,lambda_i=None)
        # Test numba
        mu1_t = solve_mu1(lhc.kappa_p,lhc.theta_p,lhc.gamma1[0], lambda_i=None)
        mu_t = compute_stationary(lhc.kappa_p,lhc.theta_p,lhc.m,lhc.gamma1[0],mu1_t,lambda_i=None)
        
        chi0 = np.append([1] , mu)
        params_actual = [lhc.kappa, lhc.theta,lhc.gamma1,lhc.lambda_i,lhc.sigma, lhc.sigma_err]
        # params_actual = np.array([0.546,0.421,
        #                           0.624,0.512,
        #                           0.205,
        #                           0,0,
        #                           0.5,0.3,
        #                           0.001])
        # lhc.unflatten_params(params_actual[:2*m+1])
        # _= lhc.build_P_params(params_actual[2*m+1:],lhc.gamma1)
        # Simulate. We are using an Euler discretization. 
        # Start at 0.5 also for aloowing for more jump op and down. Again, likly too large initial cov
        # Use same seed to reproduce same randomness.
        mat_grid = np.array([1,3,5,7,10]) # Typical maturity grid
        # mat_grid = np.array([5]) 

        n_mat = mat_grid.shape[0]
        T_path, chi_Q = lhc.simul_latent_states(chi0=chi0,T=T,M=M,n_mat=n_mat,seed=seed,
                                                scheme='Milstein',measure='P')
        # Try to simulate Z instead and see if it makes it better...
        T_path, Z_Q = lhc.simul_Z(chi0=chi0[1:],T=T,M=M,n_mat=n_mat,seed=seed,
                                                scheme='Euler',measure='P')
        Z_ones = np.hstack([np.ones((Z_Q.shape[0],1)),Z_Q])
        # Holld maturity to be 5
        t_mat_grid = np.ascontiguousarray(mat_grid[:, None] + T_path[None, :])   # shape (len(T_M_grid), len(t_obs))

        t0 = T_path

        kappa, theta, gamma1 = lhc.kappa,lhc.theta,lhc.gamma1[0]
        r = lhc.r
        Y_dim = lhc.Y_dim
        delta = lhc.delta
        tenor = lhc.tenor 
        lhc_numba = rebuild_lhc_struct(kappa, theta, gamma1, r, Y_dim, delta, tenor)


        # Draw noise vector:
        save_path = "./Simulation_studies/" 
        color_cycle = plt.cm.tab10.colors  

        R = norm.rvs(size = (t_mat_grid.shape[0]*t_mat_grid.shape[1]),scale = lhc.sigma_err).reshape(t_mat_grid.shape) # simulate at beginning - faster!
        CDS_simul_actual = get_CDS_Model(T_path, t0, t_mat_grid, chi_Q.T, lhc_numba)  
        CDS_simul = CDS_simul_actual + R # Work with small noise...
        # Test of value:
        print(cds_value(lhc_numba, t0[0], t0[0], t_mat_grid[:,0],CDS_simul[:,0]) @ chi_Q[0,:])
        print(np.min(CDS_simul),np.max(CDS_simul))
        CDS_simul_Z_actual = get_CDS_Model(T_path, t0, t_mat_grid, Z_ones.T, lhc_numba)  
        # Reestimation of the Process. Try Kalman filter on the recreated CDS spreads.
        # Kalman automatically initiates at random. 
        lhc_kalman_params,  Xn,Zn, Pn, se,ll= lhc.run_n_kalmans(T_path, t_mat_grid, CDS_simul.T,base_seed=2000,n_restarts=1)

        # Add gamma1 too...
        # lhc_kalman_params =np.concatenate([lhc_kalman_params[:2*m], lhc.gamma1, lhc_kalman_params[2*m:]])
        # same for se - Not directly clear.
        se =np.concatenate([se[:2*m],np.array([0]), se[2*m:]])

        # Xn_kalman,Yn_kalman= Xn[:,1:], Xn[:,0]
        Xn_kalman,Yn_kalman = lhc.kalman_X_Y(T_path,Xn)
        # Recalculate CDS spreads, if value approach:
        # Get CDS Spreads from kalman
        CDS_kalman = lhc.CDS_model(T_path,  t_mat_grid, CDS_simul.T , 
                                   t0=T_path, X_in=Xn_kalman.T,Y_in=Yn_kalman) 

        # FILIPOVIC SECTION . use actual cds to be 100% true to model assumption
        lhc_filipovic_params= lhc.optimize_params(T_path, t_mat_grid, CDS_simul_actual.T,base_seed=2000)
    
        # After optimize, define lhc model for inputting.
        # ---- Get states ----
        X_fil, Y_fil, Z_fil = lhc.get_states(T_path, t_mat_grid, CDS_simul.T)
        S_fil = Y_fil  # rename Y to S

        default_intensity = lhc.default_intensity(X_fil,Y_fil)

        # ---- Model CDS ----
        CDS_fil = lhc.CDS_model(T_path, t_mat_grid, CDS_simul.T)



        np.set_printoptions(precision=4, suppress=True)  # fewer decimals, no scientific notation
        print(f'Optimal parameters, Kalman: {lhc_kalman_params}')
        print(f'Optimal parameters, Filipovic: {lhc_filipovic_params}')
        print(f'Actual Parameters: {params_actual}')

        # ---- Build parameter names ----
        param_lhc_names = [f'kappa{i}' for i in range(1, X_dim+1)] \
                        + [f'theta{i}' for i in range(1, X_dim+1)] \
                        + ['gamma1'] \
                        + [f'lambda{i}' for i in range(1, X_dim+1)] \
                        + [f'sigma{i}' for i in range(1, X_dim+1)] \
                        + ['sigma_err']

        # ---- Base dataframe ----
        df_main = pd.DataFrame({
            'Parameter': param_lhc_names,
            'Estimated Kalman': lhc_kalman_params,
            'SE': se, 
            'LogLike': np.append(ll, np.zeros(lhc_kalman_params.shape[0] - 1)),
            'Filipovic': np.append(lhc_filipovic_params, np.zeros(np.array(param_lhc_names).shape[0]-lhc_filipovic_params.shape[0])),
            'True': np.concatenate(params_actual),
        })

        df_main['Abs Error, Kalman'] = np.abs(df_main['Estimated Kalman'] - df_main['True'])
        df_main['Rel Error (%), Kalman'] = 100 * np.abs(df_main['Estimated Kalman'] - df_main['True']) / df_main['True']
        df_main['Abs Error, Filipovic'] = np.abs(df_main['Filipovic'] - df_main['True'])
        df_main['Rel Error (%), Filipovic'] = 100 * np.abs(df_main['Filipovic'] - df_main['True']) / df_main['True']




        # ---- Display + save ----
        print(df_main)
        df_main.to_excel(os.path.join(save_path, f'lhc_parameter_comparison_Xdim{X_dim}.xlsx'), index=False)

        # --- Latent states plots ---
        n_states = chi_Q.shape[1]  # total number of latent states

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # ---------- Plot 1: Survival process (Y) ----------
        ax = axes[0]
        ax.plot(T_path, chi_Q[:, 0], "-", alpha=0.8, label="Survival Process", color="blue")
        ax.plot(T_path, Yn_kalman, "-.", alpha=0.9, label="Y (Kalman)", color="green")
        ax.plot(T_path, Y_fil, "--", alpha=0.9, label="Y (Filipovic)", color="yellow")

        ax.set_ylabel("Y / Survival Process")
        ax.set_title("Latent States: Y and X Dynamics")
        ax.legend(loc="best")
        ax.grid(True, linestyle="--", alpha=0.6)

        # ---------- Plot 2: X states ----------
        ax = axes[1]
        for i in range(1, n_states):
            state_name = f"X{i}"
            ax.scatter(T_path, chi_Q[:, i],  alpha=0.8, label=f"{state_name} (Simulated)", color=color_cycle[(i-1) % len(color_cycle)])
            ax.plot(T_path, X_fil[i-1, :], "-.", alpha=0.9, label=f"{state_name} (Filipovic)", color=color_cycle[(i-1) % len(color_cycle)])
            ax.plot(T_path, Xn_kalman[:, i-1], "--", alpha=0.9, label=f"{state_name} (Kalman)", color=color_cycle[(i-1) % len(color_cycle)])

        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Latent Factors X")
        ax.legend(loc="best", ncol=2)
        ax.grid(True, linestyle="--", alpha=0.6)

        fig.tight_layout()
        fig.savefig(os.path.join(save_path, f"SimulLHC_LatentStates_Combined_Xdim{X_dim}.png"), dpi=150)
        plt.close(fig)


        # --- CDS spreads plot ---
        fig, ax = plt.subplots(figsize=(10,5))

        for i in range(CDS_simul.shape[0]):
            ax.plot(T_path, CDS_simul_actual[i,:], "-", alpha=0.9, label=f"CDS, T_mat={mat_grid[i]}", color='red')
            ax.plot(T_path, CDS_simul[i,:], "-", alpha=0.1, label=f"CDS corrupted, T_mat={mat_grid[i]}", color='black')
            ax.plot(T_path, CDS_kalman[:, i], "--", alpha=0.9, label=f"CDS Kalman, T_mat={mat_grid[i]}", color='green')
            ax.plot(T_path, CDS_fil[:, i], "--", alpha=0.9, label=f"CDS Filipovic, T_mat={mat_grid[i]}", color='yellow')

        ax.set_xlabel("Time (years)")
        ax.set_ylabel("CDS Spreads")
        ax.set_title("CDS Spreads Comparison")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(save_path, f"SimulLHC_CDS_Spreads_Xdim{X_dim}.png"), dpi=150)
        plt.close(fig)


    ####### affine calibration in a true to model setting.
    T,M = 10, 1000
    # Loop over the two different X_dim comps.
    for X_dim in [1,2,3]:

        #### CIR SIMULATION. ######
        save_path = "./Simulation_studies/"  

        seed = 4000
        cir = CIR(0.00248, 0.4, 0.25,X_dim,cascading=True)
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
        # alpha = 2 * cir.kappa_p *  cir.theta_p /  cir.sigma**2
        # beta = 2 * cir.kappa_p /  cir.sigma**2
        # # CIR values.
        # lambda0 =( alpha / beta )  # start below long term (halfway there)
        lambda0 = np.cumprod(cir.theta_p[::-1])[::-1] # In the CIR cascading.

        # This is under Q 
        # T_return,lambda_eul_Q = cir.simulate_intensity(lambda0=lambda0,T=T,M=M,
        #                                                scheme="Euler",seed=seed,measure='P')
        T_return,lambda_mil_Q = cir.simulate_intensity(lambda0=lambda0,T=T,M=M,
                                                       scheme="Milstein",seed=seed,measure='P')

        mat_grid = np.array([1,3,5,7,10])
        # mat_grid = np.array([1,5])
        t_mat_grid = np.ascontiguousarray(mat_grid[:, None] + T_return[None, :])   # shape (len(T_M_grid), len(t_obs))

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

        # Save figure
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, f"True_Simul_CIR_Xdim{X_dim}.png"), dpi=150)
        plt.close(fig)
        # Gamma_kalman_noise = Gamma_kalman + R

        ## Try Estimation of Correct model
        Lambda_kalman = np.zeros(shape = (T_return.shape[0], mat_grid.shape[0]))
        for i in range(T_return.shape[0]):
            Lambda_kalman[i,:] =  - np.log(cir.Laplace_Transform(params_cir,lambda_mil_Q[i],mat_grid))
        R = norm.rvs(size = (Lambda_kalman.shape[0]*Lambda_kalman.shape[1]),
                    scale = cir.sigma_err,random_state=10).reshape(Lambda_kalman.shape) # simulate at beginning - faster!

        Lambda_kalman_noise = Lambda_kalman + R
        ## Run one kalman filter to see identification...
        color_cycle = plt.cm.tab10.colors  

        fig, (ax1) = plt.subplots(1,1, figsize=(14,6), sharey=False)

        for i in range(Lambda_kalman.shape[1]):
            color = color_cycle[i % len(color_cycle)]
            ax1.plot(T_return, Lambda_kalman_noise[:, i], "--", alpha=0.9, color=color, label=f"Actual Lambda{mat_grid[0]}")

        ax1.set_xlabel("Time (years)")
        ax1.set_ylabel("Integrated Gamma")
        ax1.legend()
        ax1.set_title("Integrated Gamma")

        fig.tight_layout()

        # Save figure
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, f"True_Lambda_gamma_fits.png"), dpi=150)
        plt.close(fig)
        

        # Select only params at maturity.
        Y_scaled = Lambda_kalman_noise 

        ll,params_cir_est, Xn_cir,Zn_cir, Pn_cir,se_cir = cir.run_kalman_filter(T_return,t_mat_grid,Y_scaled,seed=2000)
        Zn_cir = Zn_cir 
        ### illustrate fitted process.
        
        color_cycle = plt.cm.tab10.colors  

        fig, (ax1) = plt.subplots(1,1, figsize=(14,6), sharey=False)

        for i in range(Lambda_kalman.shape[1]):
            color = color_cycle[i % len(color_cycle)]
            ax1.scatter(T_return, Zn_cir[:, i], alpha=0.9, color=color, label=f"Kalman Gamma{mat_grid[0]}")
            # ax1.plot(T_return, Zn_cir_l[:, i], "--", alpha=0.9, color=color, label=f"Kalman Lambda{mat_grid[0]}")
            ax1.plot(T_return, Lambda_kalman[:, i], "--", alpha=0.9, color=color, label=f"Actual Lambda{mat_grid[0]}")

        ax1.set_xlabel("Time (years)")
        ax1.set_ylabel("Integrated Gamma")
        ax1.legend()
        ax1.set_title("Integrated Gamma")

        fig.tight_layout()

        # Save figure
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, f"True_Gamma_fits.png"), dpi=150)
        plt.close(fig)
        
        np.set_printoptions(precision=4, suppress=True)

        # Param names: 
        param_cir_names = [f'kappa{i}' for i in range(1,X_dim+1)]+ [f'theta{i}' for i in range(1,X_dim+1)]
        param_cir_names += [f'sigma{i}' for i in range(1,X_dim+1)]
        param_cir_names += [f'lambda{i}' for i in range(1,X_dim+1)]+ [f'sigma_err']
        
        df = pd.DataFrame({
            'Parameter': param_cir_names,
            'Estimated Kalman': params_cir_est,
            'True': params_cir,
            'SE': se_cir,
            'LogLike': np.append(ll, np.zeros(params_cir_est.shape[0] - 1)),

        })
        df['Abs Error'] = np.abs(df['Estimated Kalman'] - df['True'])
        df['Rel Error (%)'] = 100 * np.abs(df['Estimated Kalman'] - df['True']) / df['True']

        print(df)
        df.to_excel(os.path.join(save_path, f'True_cir_parameter_comparison_Xdim{X_dim}.xlsx'), index=False)


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

        fig.tight_layout()

        # Save figure
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, f"True_SimulCIR_states_vs_Kalman_Xdim{X_dim}.png"), dpi=150)
        plt.close(fig)


      


    ######## calibration using the actual technique.
    T,M = 10, 1000
    # Loop over the two different X_dim comps.
    for X_dim in [1,2,3]:

        #### CIR SIMULATION. ######
        save_path = "./Simulation_studies/"  

        seed = 4000
        cir = CIR(0.00248, 0.4, 0.25,X_dim,cascading=True)
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
        # alpha = 2 * cir.kappa_p *  cir.theta_p /  cir.sigma**2
        # beta = 2 * cir.kappa_p /  cir.sigma**2
        # # CIR values.
        # lambda0 =( alpha / beta )  # start below long term (halfway there)
        # lambda0 = np.cumsum(cir.theta_p[::-1])[::-1] # In the CIR cascading.
        lambda0 = np.cumprod(cir.theta_p[::-1])[::-1] # In the CIR cascading.

        # This is under Q 
        # T_return,lambda_eul_Q = cir.simulate_intensity(lambda0=lambda0,T=T,M=M,
        #                                                scheme="Euler",seed=seed,measure='P')
        T_return,lambda_mil_Q = cir.simulate_intensity(lambda0=lambda0,T=T,M=M,
                                                       scheme="Milstein",seed=seed,measure='P')

        mat_grid = np.array([1,3,5,7,10])
        # mat_grid = np.array([1,5])
        t_mat_grid = np.ascontiguousarray(mat_grid[:, None] + T_return[None, :])   # shape (len(T_M_grid), len(t_obs))

        # Get true CDS spreads. 
        CDS_cir = np.ones((t_mat_grid.T.shape))
        # CDS_cir_old = np.ones((t_mat_grid.T.shape))
        for i in range(t_mat_grid.shape[1]):
            lambda_curr  = np.array([lambda_mil_Q[i]])
            mat_curr = t_mat_grid[:,i]
            CDS_cir[i,:] = cir.cds_spread_fast(lambda_curr,params_cir,T_return[i],mat_curr)
            # CDS_cir_old[i,:] = cir.cds_spread(lambda_curr,params_cir,T_return[i],mat_curr)

            print(f'Done with {(i+1)/ t_mat_grid.shape[1]*100} CDS')

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


        ## See if possible to estimate the parameters. For that, need to turn into integrated lambda.
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
            cali_params[t_idx, : ] = model.calibrate_deterministic(CDS_cir[t_idx,:] , mat_grid, 
                                                                   0.0, t_grid_payments)
            # Generate the survial probabilities/survival process
            for i in range(extrapolate_grid.shape[0]):
                Gamma[t_idx,i] = model.Gamma_fun(cali_params[t_idx, : ],extrapolate_grid[i],t_mats)
            print(f'Done with {(t_idx+1)/ CDS_cir.shape[0]*100} CDS')
        survival = np.exp(-Gamma )


        Gamma_kalman = Gamma[:,np.isin(extrapolate_grid, mat_grid).flatten()]

        # Kalman noise.
        R = norm.rvs(size = (Gamma_kalman.shape[0]*Gamma_kalman.shape[1]),scale = cir.sigma_err,random_state=10).reshape(Gamma_kalman.shape) # simulate at beginning - faster!
        # Divide gamma process by time to mat.
        Gamma_kalman_noise = Gamma_kalman  + R
        # Gamma_kalman_noise = Gamma_kalman + R

        ## Try Estimation of Correct model
        Lambda_kalman = np.zeros(shape = (T_return.shape[0], mat_grid.shape[0]))
        for i in range(T_return.shape[0]):
            Lambda_kalman[i,:] =  - np.log(cir.Laplace_Transform(params_cir,lambda_mil_Q[i],mat_grid))
        R = norm.rvs(size = (Lambda_kalman.shape[0]*Lambda_kalman.shape[1]),
                    scale = cir.sigma_err,random_state=10).reshape(Lambda_kalman.shape) # simulate at beginning - faster!

        Lambda_kalman_noise = Lambda_kalman + R
        ## Run one kalman filter to see identification...
        color_cycle = plt.cm.tab10.colors  

        fig, (ax1) = plt.subplots(1,1, figsize=(14,6), sharey=False)

        for i in range(Lambda_kalman.shape[1]):
            color = color_cycle[i % len(color_cycle)]
            ax1.plot(T_return, Gamma_kalman[:, i], "-", alpha=0.9, color=color, label=f"Actual Gamma{mat_grid[0]}")
            ax1.plot(T_return, Lambda_kalman[:, i], "--", alpha=0.9, color=color, label=f"Actual Lambda{mat_grid[0]}")

        ax1.set_xlabel("Time (years)")
        ax1.set_ylabel("Integrated Gamma")
        ax1.legend()
        ax1.set_title("Integrated Gamma")

        fig.tight_layout()

        # Save figure
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, f"Lambda_gamma_fits.png"), dpi=150)
        plt.close(fig)
        

        # Select only params at maturity.
        Y_scaled = Gamma_kalman_noise 
        # Y_scaled = Lambda_kalman_noise 

        ll,params_cir_est, Xn_cir,Zn_cir, Pn_cir,se_cir = cir.run_kalman_filter(T_return,t_mat_grid,Y_scaled,seed=2000)
        Zn_cir = Zn_cir 
        ### illustrate fitted process.
        
        color_cycle = plt.cm.tab10.colors  

        fig, (ax1) = plt.subplots(1,1, figsize=(14,6), sharey=False)

        for i in range(Lambda_kalman.shape[1]):
            color = color_cycle[i % len(color_cycle)]
            ax1.scatter(T_return, Zn_cir[:, i], alpha=0.9, color=color, label=f"Kalman Gamma{mat_grid[0]}")
            ax1.plot(T_return, Gamma_kalman[:, i], "--", alpha=0.9, color=color, label=f"Actual Gamma{mat_grid[0]}")
            # ax1.plot(T_return, Zn_cir_l[:, i], "--", alpha=0.9, color=color, label=f"Kalman Lambda{mat_grid[0]}")
            ax1.plot(T_return, Lambda_kalman[:, i], "--", alpha=0.9, color=color, label=f"Actual Lambda{mat_grid[0]}")

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
            CDS_cir_model[i,:] = cir.cds_spread_fast(lambda_curr,params_cir_est,T_return[i],mat_curr)
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
            'SE': se_cir,
            'Log_like': np.append(ll, np.zeros(params_cir_est.shape[0] - 1)),

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

    # # SEction to investigate misspecification.


##### APPENDIX TABLE: TESTING IF POSSIBLE TO RETRIEVE TRUS PARAMETERS IN THE UNCONSTRAINED MODEL.
    from Models.LHCModels.LHC_wGamma1 import LHC_single as LHC_gamma1
    from Models.LHCModels.LHC_wGamma1 import get_CDS_Model,kalmanfilter_opt, calc_gamma1,rebuild_lhc_struct, cds_value, solve_mu1, compute_stationary
    # Also some realistic size (TODO)
    # Simulat eover longer time horizon
    T,M = 10, 1000
    # # Loop over the two different X_dim comps.
    for X_dim in [1,2,3]: #,2,3]:

        ### Simulate 1 LHC dataset with specific parameter Choises.
        lhc = LHC_gamma1(0.00248,0.4,0.25)
        Y_dim,m = 1,X_dim
        # Here, parameters are set already
        seed = 4000
        rng = np.random.default_rng(seed)
        X0 = 0.3
        chi0 = np.array([1] + [X0]*m)
        lhc.initialise_LHC(Y_dim,m,X0=X0,rng=rng)

        lhc.flatten_params()
        params = lhc.flatten_params()
        lhc.unflatten_params(params[:2*m+1])
         ### Just for fairness, initialise at mu1 for z
        # Set also P params. 
        lhc_P = lhc.build_P_params(rng=rng)
        # Start value stationary ones. 
        mu1 = lhc.solve_mu1(lhc.kappa_p,lhc.theta_p,lhc.gamma1, lambda_i=None)
        mu = lhc.compute_stationary(lhc.kappa_p,lhc.theta_p,lhc.m,lhc.gamma1,mu1,lambda_i=None)
        # Test numba
        mu1_t = solve_mu1(lhc.kappa_p,lhc.theta_p,lhc.gamma1[0], lambda_i=None)
        mu_t = compute_stationary(lhc.kappa_p,lhc.theta_p,lhc.m,lhc.gamma1[0],mu1_t,lambda_i=None)
        
        chi0 = np.append([1] , mu)
        params_actual = [lhc.kappa, lhc.theta,lhc.gamma1,lhc.lambda_i,lhc.sigma, lhc.sigma_err]
        # params_actual = np.array([0.546,0.421,
        #                           0.624,0.512,
        #                           0.205,
        #                           0,0,
        #                           0.5,0.3,
        #                           0.001])
        # lhc.unflatten_params(params_actual[:2*m+1])
        # _= lhc.build_P_params(params_actual[2*m+1:],lhc.gamma1)
        # Simulate. We are using an Euler discretization. 
        # Start at 0.5 also for aloowing for more jump op and down. Again, likly too large initial cov
        # Use same seed to reproduce same randomness.
        mat_grid = np.array([1,3,5,7,10]) # Typical maturity grid
        # mat_grid = np.array([5]) 

        n_mat = mat_grid.shape[0]
        T_path, chi_Q = lhc.simul_latent_states(chi0=chi0,T=T,M=M,n_mat=n_mat,seed=seed,
                                                scheme='Milstein',measure='P')
        # Try to simulate Z instead and see if it makes it better...
        T_path, Z_Q = lhc.simul_Z(chi0=chi0[1:],T=T,M=M,n_mat=n_mat,seed=seed,
                                                scheme='Euler',measure='P')
        Z_ones = np.hstack([np.ones((Z_Q.shape[0],1)),Z_Q])
        # Holld maturity to be 5
        t_mat_grid = np.ascontiguousarray(mat_grid[:, None] + T_path[None, :])   # shape (len(T_M_grid), len(t_obs))

        t0 = T_path

        kappa, theta, gamma1 = lhc.kappa,lhc.theta,lhc.gamma1[0]
        r = lhc.r
        Y_dim = lhc.Y_dim
        delta = lhc.delta
        tenor = lhc.tenor 
        lhc_numba = rebuild_lhc_struct(kappa, theta, gamma1, r, Y_dim, delta, tenor)


        # Draw noise vector:
        save_path = "./Simulation_studies/" 
        color_cycle = plt.cm.tab10.colors  

        R = norm.rvs(size = (t_mat_grid.shape[0]*t_mat_grid.shape[1]),scale = lhc.sigma_err).reshape(t_mat_grid.shape) # simulate at beginning - faster!
        CDS_simul_actual = get_CDS_Model(T_path, t0, t_mat_grid, chi_Q.T, lhc_numba)  
        CDS_simul = CDS_simul_actual + R # Work with small noise...
        # Test of value:
        print(cds_value(lhc_numba, t0[0], t0[0], t_mat_grid[:,0],CDS_simul[:,0]) @ chi_Q[0,:])
        print(np.min(CDS_simul),np.max(CDS_simul))
        CDS_simul_Z_actual = get_CDS_Model(T_path, t0, t_mat_grid, Z_ones.T, lhc_numba)  
        # Reestimation of the Process. Try Kalman filter on the recreated CDS spreads.
        # Kalman automatically initiates at random. 
        lhc_kalman_params,  Xn,Zn, Pn, se,ll= lhc.run_n_kalmans(T_path, t_mat_grid, CDS_simul.T,base_seed=2000,n_restarts=1)
        # Xn_kalman,Yn_kalman= Xn[:,1:], Xn[:,0]
        Xn_kalman,Yn_kalman = lhc.kalman_X_Y(T_path,Xn)
        # Recalculate CDS spreads, if value approach:
        # Get CDS Spreads from kalman
        CDS_kalman = lhc.CDS_model(T_path,  t_mat_grid, CDS_simul.T , 
                                   t0=T_path, X_in=Xn_kalman.T,Y_in=Yn_kalman) 

        # FILIPOVIC SECTION . use actual cds to be 100% true to model assumption
        lhc_filipovic_params= lhc.optimize_params(T_path, t_mat_grid, CDS_simul_actual.T,base_seed=2000)
    
        # After optimize, define lhc model for inputting.
        # ---- Get states ----
        X_fil, Y_fil, Z_fil = lhc.get_states(T_path, t_mat_grid, CDS_simul.T)
        S_fil = Y_fil  # rename Y to S

        default_intensity = lhc.default_intensity(X_fil,Y_fil)

        # ---- Model CDS ----
        CDS_fil = lhc.CDS_model(T_path, t_mat_grid, CDS_simul.T)



        np.set_printoptions(precision=4, suppress=True)  # fewer decimals, no scientific notation
        print(f'Optimal parameters, Kalman: {lhc_kalman_params}')
        print(f'Optimal parameters, Filipovic: {lhc_filipovic_params}')
        print(f'Actual Parameters: {params_actual}')

        # ---- Build parameter names ----
        param_lhc_names = [f'kappa{i}' for i in range(1, X_dim+1)] \
                        + [f'theta{i}' for i in range(1, X_dim+1)] \
                        + ['gamma1'] \
                        + [f'lambda{i}' for i in range(1, X_dim+1)] \
                        + [f'sigma{i}' for i in range(1, X_dim+1)] \
                        + ['sigma_err']

        # ---- Base dataframe ----
        df_main = pd.DataFrame({
            'Parameter': param_lhc_names,
            'Estimated Kalman': lhc_kalman_params,
            'SE': se, 
            'LogLike': np.append(ll, np.zeros(lhc_kalman_params.shape[0] - 1)),
            'Filipovic': np.append(lhc_filipovic_params, np.zeros(np.array(param_lhc_names).shape[0]-lhc_filipovic_params.shape[0])),
            'True': np.concatenate(params_actual),
        })

        df_main['Abs Error, Kalman'] = np.abs(df_main['Estimated Kalman'] - df_main['True'])
        df_main['Rel Error (%), Kalman'] = 100 * np.abs(df_main['Estimated Kalman'] - df_main['True']) / df_main['True']
        df_main['Abs Error, Filipovic'] = np.abs(df_main['Filipovic'] - df_main['True'])
        df_main['Rel Error (%), Filipovic'] = 100 * np.abs(df_main['Filipovic'] - df_main['True']) / df_main['True']




        # ---- Display + save ----
        print(df_main)
        df_main.to_excel(os.path.join(save_path, f'lhc_parameter_comparison_Xdim{X_dim}_wgamma1.xlsx'), index=False)

        # --- Latent states plots ---
        n_states = chi_Q.shape[1]  # total number of latent states

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # ---------- Plot 1: Survival process (Y) ----------
        ax = axes[0]
        ax.plot(T_path, chi_Q[:, 0], "-", alpha=0.8, label="Survival Process", color="blue")
        ax.plot(T_path, Yn_kalman, "-.", alpha=0.9, label="Y (Kalman)", color="green")
        ax.plot(T_path, Y_fil, "--", alpha=0.9, label="Y (Filipovic)", color="yellow")

        ax.set_ylabel("Y / Survival Process")
        ax.set_title("Latent States: Y and X Dynamics")
        ax.legend(loc="best")
        ax.grid(True, linestyle="--", alpha=0.6)

        # ---------- Plot 2: X states ----------
        ax = axes[1]
        for i in range(1, n_states):
            state_name = f"X{i}"
            ax.scatter(T_path, chi_Q[:, i],  alpha=0.8, label=f"{state_name} (Simulated)", color=color_cycle[(i-1) % len(color_cycle)])
            ax.plot(T_path, X_fil[i-1, :], "-.", alpha=0.9, label=f"{state_name} (Filipovic)", color=color_cycle[(i-1) % len(color_cycle)])
            ax.plot(T_path, Xn_kalman[:, i-1], "--", alpha=0.9, label=f"{state_name} (Kalman)", color=color_cycle[(i-1) % len(color_cycle)])

        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Latent Factors X")
        ax.legend(loc="best", ncol=2)
        ax.grid(True, linestyle="--", alpha=0.6)

        fig.tight_layout()
        fig.savefig(os.path.join(save_path, f"SimulLHC_LatentStates_Combined_Xdim{X_dim}_wgamma1.png"), dpi=150)
        plt.close(fig)


        # --- CDS spreads plot ---
        fig, ax = plt.subplots(figsize=(10,5))

        for i in range(CDS_simul.shape[0]):
            ax.plot(T_path, CDS_simul_actual[i,:], "-", alpha=0.9, label=f"CDS, T_mat={mat_grid[i]}", color='red')
            ax.plot(T_path, CDS_simul[i,:], "-", alpha=0.1, label=f"CDS corrupted, T_mat={mat_grid[i]}", color='black')
            ax.plot(T_path, CDS_kalman[:, i], "--", alpha=0.9, label=f"CDS Kalman, T_mat={mat_grid[i]}", color='green')
            ax.plot(T_path, CDS_fil[:, i], "--", alpha=0.9, label=f"CDS Filipovic, T_mat={mat_grid[i]}", color='yellow')

        ax.set_xlabel("Time (years)")
        ax.set_ylabel("CDS Spreads")
        ax.set_title("CDS Spreads Comparison")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(save_path, f"SimulLHC_CDS_Spreads_Xdim{X_dim}_wgamma1.png"), dpi=150)
        plt.close(fig)



    test = 1