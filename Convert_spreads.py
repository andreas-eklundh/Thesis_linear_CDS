from Models.BaselineCIR_alternative.Gamma_solver import DeterministicGamma
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.dates as mdates


if __name__ == "__main__":
    # parameters
    r = 0.00248
    delta = 0.4
    tenor = 0.25

    #### Now callibrate and store each rate to get eficiently a surface:
    #### Preliminary investigation. 
    sub_df = pd.read_excel("./Data/subset_data.xlsx")
    firms = ['CMZB','DANBNK','MONTE', 'SVSKHB']
    # firms = ['MONTE']

    # Loop over each firm in list.
    for firm in firms:
        test_df = sub_df[(sub_df['Ticker']==firm)]
        test_df = test_df.pivot(index = ['Date','Ticker'],
                                columns='Tenor',values = 'Par Spread').reset_index()
        # Test on subset data ownly to get very few obs. One large spread increase to test.
        test_df['Years']= ((test_df['Date'] - test_df['Date'].min()).dt.total_seconds() / (365.25 * 24 * 3600)).drop_duplicates()

        t = np.array(test_df['Years'])

        mat_grid = np.array([1,2,3,4,5,7,10])
        # mat_grid = np.array([5])
        t0 = 0.0
        t_mats = np.concatenate(([t0], mat_grid))

        t_mat_grid = np.ascontiguousarray(t_mats[:, None] + t[None, :])   # shape (len(T_M_grid), len(t_obs))

        # For simplicity just assume t0=t. 
        # Payments are then every 0.25 year.     

        # Get payment grids of quarterly.
        # CDS_obs = np.array(test_df[['1Y','3Y','5Y','7Y','10Y']].ffill().bfill())
        CDS_obs = np.array(test_df[['1Y','2Y','3Y','4Y','5Y','7Y','10Y']].ffill().bfill())

        # CDS_obs = np.array(test_df[['5Y']].ffill().bfill())
        model = DeterministicGamma(r, delta, tenor)

        # RUN LATER - MORE CONSUMING
        # Generate synthetic "market" CDS values (set to zero at par spread condition)
        
        # To have some grid to plot.
        plot_grid = np.array([i *0.2 for i in range(int(np.max(mat_grid)/0.2)+ 1)])    
        Gamma, survival = np.zeros((CDS_obs.shape[0], plot_grid.shape[0])),np.zeros((CDS_obs.shape[0], plot_grid.shape[0]))
        cali_params = np.zeros((CDS_obs.shape[0], mat_grid.shape[0]))
        for t_idx in range(CDS_obs.shape[0]):
            # Loop over time points
            # Generate necessary values:        
            # Calibrate back hazard rates
            t_grid_payments = np.array([tenor*i for i in range(int(np.max(mat_grid)/tenor)+1)])   
            
            cali_params[t_idx, : ] = model.calibrate_deterministic(CDS_obs[t_idx,:] , mat_grid, 0.0, t_grid_payments)
            # Generate the survial probabilities/survival process
            for i in range(plot_grid.shape[0]):
                # Get integrated process for each plot grid points. Is over maturities.
                Gamma[t_idx,i] = model.Gamma_fun(cali_params[t_idx, : ] , 
                                                 plot_grid[i],t_mats)
                
            print(f'Done implied {(t_idx+1)/CDS_obs.shape[0]}, {firm}')
        survival = np.exp(-Gamma )

        save_path = f"./Gamma_Calibration/{firm}/" 
        os.makedirs(save_path, exist_ok=True)

        # First save processes for use in later stuff.
        np.savez(os.path.join(save_path, f"Data_{firm}.npz"),
             t_mats_plots = plot_grid,
             survival=survival,
             Gamma = Gamma,
             default_prob = 1- survival,
             gamma_hist = cali_params)
        
    for firm in firms:
        test_df = sub_df[(sub_df['Ticker']==firm)]
        test_df = test_df.pivot(index = ['Date','Ticker'],
                                columns='Tenor',values = 'Par Spread').reset_index()
        # Test on subset data ownly to get very few obs. One large spread increase to test.
        test_df['Years']= ((test_df['Date'] - test_df['Date'].min()).dt.total_seconds() / (365.25 * 24 * 3600)).drop_duplicates()

        t = np.array(test_df['Years'])

        mat_grid = np.array([1,2,3,4,5,7,10])
        t0 = 0.0
        t_mats = np.concatenate(([t0], mat_grid))

        t_mat_grid = np.ascontiguousarray(mat_grid[:, None] + t[None, :])   # shape (len(T_M_grid), len(t_obs))

        # For simplicity just assume t0=t. 
        # Payments are then every 0.25 year.     

        # Get payment grids of quarterly.
        CDS_obs = np.array(test_df[['1Y','2Y','3Y','4Y','5Y','7Y','10Y']].ffill().bfill())
        model = DeterministicGamma(r, delta, tenor)

        # RUN LATER - MORE CONSUMING
        # Generate synthetic "market" CDS values (set to zero at par spread condition)
        
        # To have some grid to plot.
        plot_grid = np.array([i *0.2 for i in range(int(np.max(mat_grid)/0.2)+ 1)])    
        #### Break look so above is just a 'oneoff'
        data = np.load(f"./Gamma_Calibration/{firm}/Data_{firm}.npz")
        t_mats_plots = data['t_mats_plots']
        survival=data['survival']
        Gamma = data['Gamma']
        default_prob = data['default_prob']
        gammas = data['gamma_hist']

        save_path = f"./Gamma_Calibration/{firm}/"   
        os.makedirs(save_path, exist_ok=True)
        # 3D plot.
        # Create meshgrid for 3D plotting
        dates_num = mdates.date2num(test_df['Date'])

        # Create meshgrid with numeric dates
        T, M = np.meshgrid(dates_num, plot_grid, indexing='ij')  # (n_time, n_maturities)

        # Plot
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(T, M, 1 - np.exp(-Gamma), cmap='viridis', edgecolor='k', alpha=0.9)
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # Label axes
        ax.set_xlabel('Time t')
        ax.set_ylabel('Maturity t_M')
        ax.set_zlabel('Gamma(t,t_M)')
        ax.set_title('Cumulative Hazard Surface')

        fig.colorbar(surf, shrink=0.5, aspect=10, label='Gamma')
        fig.savefig(os.path.join(save_path, f"Default_curve_{firm}.png"), dpi=150)
        plt.close(fig)


        ### Plot of integrated default probs for maturities.
        # Then plot to Check if we reproduce obs. 
        # Restric gamma:
        t_mats_plots_sub = t_mats_plots[np.isin(t_mats_plots,mat_grid).flatten()]

        survival_sub = survival[:,np.isin(t_mats_plots, mat_grid).flatten()]
        Gamma_sub = Gamma[:,np.isin(t_mats_plots, mat_grid).flatten()]
        plt.figure(figsize=(12, 6))
        
        for j, T in enumerate(mat_grid):
            # Implied line
            plt.plot(t, (1 - np.exp(-Gamma_sub))[:, j], 
                    label=f"{T}Y", linestyle='--')

        plt.xlabel("Time")
        plt.ylabel("Implied Default probability")
        plt.title("Implied Default probability by Maturity")
        plt.legend(ncol=2)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"Implied_default_{firm}.png"), dpi=150)
        plt.close(fig)

        plt.figure(figsize=(12, 6))
        
        for j, T in enumerate(mat_grid):
            # Implied line
            plt.plot(t, survival_sub[:, j], 
                    label=f"{T}Y", linestyle='--')

        plt.xlabel("Time")
        plt.ylabel("Implies Survival probability")
        plt.title("Implied Survival probability by Maturity")
        plt.legend(ncol=2)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"Implied_survival_{firm}.png"), dpi=150)
        plt.close(fig)


        # Try and generate CDS plot. 
        CDS_implied = np.zeros((CDS_obs.shape[0], t_mats.shape[0]))

        for t_idx in range(CDS_obs.shape[0]):
            # Generate necessary values:        
            # Calibrate back hazard rates
            t_grid_payments = np.array([tenor*i for i in range(int(np.max(mat_grid)/tenor)+1)])   

            for t_mat in range(len(mat_grid)):
                I1, I2, prot = model.get_CDS_deterministic(0.0,0.0,mat_grid[t_mat],
                                                                    gammas[t_idx,:],t_grid_payments,t_mats) 

                CDS_implied[t_idx,t_mat] = prot / (I1+I2)
            print(f'Done with CDS Construction {(t_idx+1)/CDS_obs.shape[0]}, {firm}')

        # Then plot to Check if we reproduce obs. 
        plt.figure(figsize=(12, 6))

        for j, T in enumerate(mat_grid):
            # Observed line
            plt.plot(t, CDS_obs[:, j], 
                    label=f"Obs {T}Y", marker='o', alpha=0.1)
            # Implied line
            plt.plot(t, CDS_implied[:, j], 
                    label=f"Implied {T}Y", linestyle='--')

        plt.xlabel("Time")
        plt.ylabel("CDS Spread (bps)")
        plt.title("Observed vs Implied CDS Spreads by Maturity")
        plt.legend(ncol=2)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"Reconstructed_Spreads__{firm}.png"), dpi=150)
        plt.close(fig)


        print(f'Done with {firm}')