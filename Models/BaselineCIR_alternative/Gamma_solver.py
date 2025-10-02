import numpy as np 
from scipy.integrate import quad
from numba.experimental import jitclass 
from scipy.optimize import brentq
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize, NonlinearConstraint
import matplotlib.dates as mdates


def _get_default_grid(u, t_grid):
    """Return time since previous payment date (in same time units as t_grid).
       If u <= first point, return 0.0.
       t_grid must be sorted and include the start time t0.
    """
    if u <= t_grid[0]:
        return 0.0
    for idx in range(len(t_grid) - 1):
        if (u > t_grid[idx]) and (u <= t_grid[idx + 1]):
            return (u - t_grid[idx])
    # if beyond last payment date:
    return max(0.0, u - t_grid[-1])

class DeterministicGamma:
    def __init__(self, r, delta, tenor=1.0):
        self.r = float(r)
        self.delta = float(delta)
        self.tenor = tenor

    def gamma_fun(self, params, u, t_mats):
        """Piecewise constant gamma(u). params length == len(t_mats)-1."""
        t_mats = np.asarray(t_mats)
        # if before first grid point
        if u < t_mats[0]:
            return 0.0
        # find index j such that t_mats[j] <= u < t_mats[j+1]
        idx = np.searchsorted(t_mats, u, side='right') - 1
        if idx < 0:
            return 0.0
        if idx >= len(params):
            # if u beyond last interval, return last param
            return params[-1]
        return params[idx]

    def Gamma_fun(self, params, u, t_mats):
        """Cumulative hazard up to u for piecewise-constant params.
           params length == len(t_mats)-1, t_mats sorted.
        """
        Gamma = 0.0
        for j in range(len(params)):
            left = t_mats[j]
            # if not last interval, right is next grid; else right is u (allow growth beyond last grid)
            right = t_mats[j+1] if (j+1) < len(t_mats) else np.inf
            if u <= left:
                break
            delta = min(u, right) - left
            if delta > 0:
                Gamma += params[j] * delta
            if u <= right:
                break
        return Gamma

    def get_CDS_deterministic(self, t, t0, t_M, params, t_grid_payments, t_mats):
        t_grid = np.asarray(t_grid_payments)
        # DISCRETE PREMIUM LEG (coupon dates)
        I1 = 0.0
        for j in range(1, len(t_grid)):
            pay = t_grid[j]
            if pay > t_M:
                break
            delta = t_grid[j] - t_grid[j-1]
            survival = np.exp(-(self.Gamma_fun(params, pay, t_mats) - self.Gamma_fun(params, t, t_mats)))
            disc = np.exp(-self.r * (pay - t))
            I1 += delta * disc * survival

        # ACCRUED PREMIUM (expected accrued coupon at default)
        def I2_integrand(u):
            accrual = _get_default_grid(u, t_grid)  # time since last coupon (in years)
            disc = np.exp(-self.r * (u - t))
            survival = np.exp(-(self.Gamma_fun(params, u, t_mats) - self.Gamma_fun(params, t, t_mats)))
            gamma_u = self.gamma_fun(params, u, t_mats)
            return accrual * disc * survival * gamma_u

        I2, _ = quad(I2_integrand, t0, t_M, epsabs=1e-9, epsrel=1e-9)

        # PROTECTION LEG
        def prot_integrand(u):
            disc = np.exp(-self.r * (u - t))
            survival = np.exp(-(self.Gamma_fun(params, u, t_mats) - self.Gamma_fun(params, t, t_mats)))
            gamma_u = self.gamma_fun(params, u, t_mats)
            return (1.0 - self.delta) * disc * survival * gamma_u

        prot, _ = quad(prot_integrand, t0, t_M, epsabs=1e-9, epsrel=1e-9)

        return I1, I2, prot

    def calibrate_deterministic(self, cds_obs, maturities, t0, t_grid_payments, tol=1e-8):
        """
        cds_obs: array of par spreads in DECIMAL (e.g. 0.01 = 100 bps)
        maturities: array of maturities [T1, T2, ...] (scalar years)
        t0: scalar start time (0.0)
        t_grid_payments: payment dates array (must include t0 and cover up to max maturity)
        Returns: params (lambdas)
        """
        params = []
        t_mats = np.concatenate(([t0], maturities))
        n = len(maturities)
        for j, T in enumerate(maturities):
            def f(lam):
                # trial params: existing + lam for this interval, zeros afterwards
                trial = params + [lam] + [0.0] * (n - j - 1)
                I1, I2, prot = self.get_CDS_deterministic(t0, t0, T, trial, t_grid_payments, t_mats)
                return cds_obs[j] * (I1 + I2) - prot

            # bracket search for brentq: expand upper bound until sign change or limit
            a = 1e-10
            b = 0.5
            fa = f(a)
            fb = f(b)
            max_b = 1e3
            while fa * fb > 0 and b < max_b:
                b *= 2.0
                fb = f(b)
            if fa * fb > 0:
                raise RuntimeError(f"No sign change for root finding for maturity {T}: f(a)={fa}, f(b)={fb}")
            lam_j = brentq(f, a, b, xtol=tol)
            params.append(lam_j)
        return np.array(params)

    def cir_solution(self, cir_params,T,rho=1):
        # Local copies of kappa, theta to minimize code. Rename to comply with Lando.
        kappa,mu,sigma1,x0 = cir_params[0], cir_params[1],  cir_params[2], cir_params[3] 
        
        gamma = np.sqrt(kappa**2 + 2*sigma1**2*rho)

        beta_nom = (- 2 * rho * (np.exp(gamma*T)-1) + 
                        x0 * np.exp(gamma * T) * (gamma - kappa) + 
                        x0 * (gamma + kappa))
        
        beta_denom = (2 * gamma + 
                        (gamma + kappa - x0 * sigma1**2) * (np.exp(gamma * T) - 1))

        beta = beta_nom / beta_denom

        alpha_log_nom = (2 * gamma * np.exp((gamma + kappa ) * T / 2))

        alpha_log_denom = (2 * gamma + 
                            (gamma + kappa - x0 * sigma1**2)*(np.exp(gamma * T)-1))
        alpha = (2 * kappa * mu * 
                    np.log(alpha_log_nom/alpha_log_denom)
                    / sigma1**2) 
        
        return alpha,beta

    # compute time T deriative - same as using T-t as parametriazation.  
    def cir_solution_T(self,cir_params,T,rho=1):
        kappa,mu,sigma1,x0 = cir_params[0], cir_params[1],  cir_params[2] , cir_params[3]
        
        gamma = np.sqrt(kappa**2 + 2*sigma1**2*rho)
        beta_nom1 =  (- 2 * rho * gamma*(np.exp(gamma*T)) + 
                        gamma*x0 * np.exp(gamma * T) * (gamma - kappa))

        beta_denom = (2 * gamma + 
                        (gamma + kappa - x0 * sigma1**2) * (np.exp(gamma * T) - 1))

        
        term1 = beta_nom1 / beta_denom

        beta_nom2 = (- 2 * rho * (np.exp(gamma*T)-1) + 
                        x0 * np.exp(gamma * T) * (gamma - kappa) + 
                        x0 * (gamma + kappa)) *  (gamma + kappa - x0 * sigma1**2) *gamma* (np.exp(gamma * T)) 
        beta_denom2 = (2 * gamma + 
                        (gamma + kappa - x0 * sigma1**2) * (np.exp(gamma * T) - 1))**2

        term2 = beta_nom2 / beta_denom2

        beta_T = term1 - term2

        # get alpha, alo verify above. 
        alpha_log_nom = (2 * gamma * np.exp((gamma + kappa ) * T / 2))

        alpha_log_denom = (2 * gamma + 
                            (gamma + kappa - x0 * sigma1**2)*(np.exp(gamma * T)-1))
        
        alpha_T = 2 * kappa * mu / sigma1**2 * (
            alpha_log_denom/alpha_log_nom * (
                gamma * (gamma + kappa) / alpha_log_denom - 
                alpha_log_nom * (gamma + kappa - x0*sigma1**2) * gamma * np.exp(gamma*T) /alpha_log_denom
            )
        )
                    
        return alpha_T, beta_T

    #### CIR/stochastic calibration. 
    # CIR ZCB/Pricing. 
    def Laplace_Transform(self,cir_params,lambda_t, T):
        alpha,beta = self.cir_solution(cir_params,T,rho=1)        
        return np.exp(alpha + beta @ lambda_t)[0]
    

    def psi_func(self,cir_params, params, u, t_mats):
        kappa,mu,sigma1,x0 = cir_params[0], cir_params[1],  cir_params[2] , cir_params[3]

        # here params is actually gamma. 

        gamma_curr = self.gamma_fun(params,u,t_mats)
        # P_cir = self.Laplace_Transform(cir_params, lambda0, 0, u)
        alpha_T, beta_T = self.cir_solution_T(cir_params,T=u)

        cir_deriv = alpha_T +  beta_T * x0 

        # Need to determine the derivatives analytically. 
        psi = gamma_curr + cir_deriv
        return psi

    def objective(self, CIR_params, params, t_mats):
        t_M_max = np.max(t_mats)

        psi_func = lambda u: self.psi_func(CIR_params,params, u, t_mats)
        integrand = lambda u: psi_func(u)**2
        t0=0.0
        PSI_obj, _ = quad(integrand, t0, t_M_max, epsabs=1e-9, epsrel=1e-9)

        return PSI_obj

    def optimize_CIR(self, CIR_params_init, gamma_params, t_mats):
        grid = np.linspace(0.0, np.max(t_mats), 20)  # 20 test points

        def constraint_fun(CIR_params):
            return np.array([self.psi_func(CIR_params, gamma_params, u, t_mats) for u in grid])

        # NonlinearConstraint ensures all values >= 0
        cons = NonlinearConstraint(constraint_fun, 0, np.inf)
                
        # Feller condition: 2*kappa*theta - sigma^2 >= 0
        def feller_constraint(CIR_params):
            kappa, theta, sigma, x0 = CIR_params
            return 2 * kappa * theta - sigma**2

        cons2 = NonlinearConstraint(feller_constraint, 0, np.inf)

        result = minimize(
            self.objective,
            CIR_params_init,
            args=(gamma_params,t_mats),
            method="trust-constr",  # works well for bounded problems
            bounds=[(1e-6, 10), (1e-6, 1), (1e-6, 2), (1e-6, 1)],  # example bounds
            constraints=[cons,
                         cons2]
            )
        
        return result


if __name__ == "__main__":
    # parameters
    r = 0.0252
    delta = 0.4
    tenor = 0.25
    model = DeterministicGamma(r, delta, tenor)

    # true lambdas (piecewise-constant over intervals [0,1],[1,2],[2,3],[3,4])
    true_lambdas = [0.02, 0.03, 0.05, 0.06]
    maturities = np.array([1.0, 2.0, 3.0, 4.0])  # calibration maturities
    t0 = 0.0
    # payment dates: yearly payments from 0 to 4
    t_grid_payments = np.arange(0.0, 4.0 + tenor, tenor)  # [0,1,2,3,4]
    # t_mats (partition for lambdas)
    t_mats = np.concatenate(([t0], maturities))

    # compute par spreads from true lambdas
    cds_test = []
    for T in maturities:
        I1, I2, prot = model.get_CDS_deterministic(t0, t0, T, true_lambdas, t_grid_payments, t_mats)
        R = prot / (I1 + I2)  # par spread in DECIMAL
        cds_test.append(R)
    cds_test = np.array(cds_test)

    # calibrate (should recover almost the true_lambdas)
    calibrated = model.calibrate_deterministic(cds_test, maturities, t0, t_grid_payments)
    print("true_lambdas   :", true_lambdas)
    print("calibrated     :", calibrated)


    #### Preliminary investigation. 
    test_df = pd.read_excel("./Data/test_data.xlsx")
    test_df = test_df.pivot(index = ['Date','Ticker'],
                            columns='Tenor',values = 'Par Spread').reset_index()
    # Test on subset data ownly to get very few obs. One large spread increase to test.
    # Callibrate on first day
    test_df = test_df[test_df.index == 0]
    test_df['Years'] = ((test_df['Date'] - test_df['Date'].min()).dt.total_seconds() / (365.25 * 24 * 3600)).drop_duplicates()
    t = np.array(test_df['Years'])
    mat_grid = np.array([1,3,5,7,10])

    t_mat_grid = np.ascontiguousarray(mat_grid[:, None] + t[None, :])   # shape (len(T_M_grid), len(t_obs))
    CDS_obs = np.array(test_df[['1Y','3Y','5Y','7Y','10Y']]).flatten()


    model = DeterministicGamma(r, delta, tenor)

    # Generate synthetic "market" CDS values (set to zero at par spread condition)

    # Generate necessary values:
    t_mats = np.concatenate(([t0], mat_grid))
    t_grid_payments = np.arange(0.0, np.max(t_mats) + tenor, tenor)  # [0,1,2,3,4]

    # Calibrate back hazard rates
    cali_params_single = model.calibrate_deterministic(CDS_obs , mat_grid, t0, t_grid_payments)

    print(f'Callibrated Gamma {cali_params_single}, date {test_df["Date"][0]} ')


    # Generate the survial probabilities/survival process
    # Time series.

    plot_grid = np.array([i *0.01 for i in range(int(10/0.01)+ 1)])    
    print(plot_grid)
    Gamma, survival = np.zeros(plot_grid.shape[0]),np.zeros(plot_grid.shape[0])
    for i in range(survival.shape[0]):
        Gamma[i] = model.Gamma_fun(cali_params_single,plot_grid[i],t_mats)
        survival[i] = np.exp(-Gamma[i] )
        

    save_path = "./Gamma_Calibration/"   # <--- change to your path
    os.makedirs(save_path, exist_ok=True)


    plt.plot(plot_grid, survival, label="Survival Probability")
    plt.xlabel("Time")
    plt.ylabel("S(t)")
    plt.title("Survival Probability Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, f"Survival_prop_{test_df["Date"].iloc[0]._date_repr}.png"), dpi=150)

    plt.close()


    # plot approximation by Gamma(t)

    plt.plot(plot_grid, 1 - survival, label=r"$\mathbb{Q}(\tau < t) = 1 - \exp(-\Gamma(t))$")

    plt.plot(plot_grid, Gamma, label=r"$\mathbb{Q}(\tau < t) \approx \Gamma(t)$", linestyle='--')

    plt.xlabel("Time")
    plt.ylabel(r"$\mathbb{Q}(\tau < t)$")
    plt.title("Default Probability Curves")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_path, f"Default_prop_{test_df["Date"].iloc[0]._date_repr}.png"), dpi=150)

    plt.close()



    #### Now callibrate and store each rate to get eficiently a surface:
    test_df = pd.read_excel("./Data/test_data.xlsx")
    test_df = test_df.pivot(index = ['Date','Ticker'],
                            columns='Tenor',values = 'Par Spread').reset_index()
    # Test on subset data ownly to get very few obs. One large spread increase to test.
    # test_df = test_df[::5]
    test_df['Years'] = ((test_df['Date'] - test_df['Date'].min()).dt.total_seconds() / (365.25 * 24 * 3600)).drop_duplicates()
    t = np.array(test_df['Years'])
    mat_grid = np.array([1,3,5,7,10])

    t_mat_grid = np.ascontiguousarray(mat_grid[:, None] + t[None, :])   # shape (len(T_M_grid), len(t_obs))

    # For simplicity just assume t0=t. 
    # Payments are then every 0.25 year.     

    # Get payment grids of quarterly.
    CDS_obs = np.array(test_df[['1Y','3Y','5Y','7Y','10Y']])
    model = DeterministicGamma(r, delta, tenor)

    # RUN LATER - MORE CONSUMING
    # Generate synthetic "market" CDS values (set to zero at par spread condition)
    
    # To have some grid to plot.
    plot_grid = np.array([i *0.1 for i in range(int(np.max(mat_grid)/0.1)+ 1)])    
    # Gamma, survival = np.zeros((CDS_obs.shape[0], plot_grid.shape[0])),np.zeros((CDS_obs.shape[0], plot_grid.shape[0]))
    # cali_params = np.zeros((CDS_obs.shape[0], mat_grid.shape[0]))
    # for t_idx in range(CDS_obs.shape[0]):
    #     # Generate necessary values:        
    #     # Calibrate back hazard rates
    #     t_grid_payments = np.array([tenor*i for i in range(int(np.max(mat_grid)/tenor)+1)])   
        
    #     cali_params[t_idx, : ] = model.calibrate_deterministic(CDS_obs[t_idx,:] , mat_grid, 0.0, t_grid_payments)
    #     # Generate the survial probabilities/survival process
    #     for i in range(plot_grid.shape[0]):
    #         Gamma[t_idx,i] = model.Gamma_fun(cali_params[t_idx, : ] ,plot_grid[i],t_mats)
    #         survival[t_idx,i] = np.exp(-Gamma[t_idx,i] )
            
    #     print(f'Done with date {t_idx+1} out of {CDS_obs.shape[0]}')

    # # First save processes for use in later stuff.
    # np.savez(os.path.join(save_path, f"CDS_TS_plot.npz"),
    #      t_mats_plots = plot_grid,
    #      survival=survival,
    #      Gamma = Gamma,
    #      default_prob = 1- survival,
    #      gamma_hist = cali_params)
    
    data = np.load("C:/Users/andre/OneDrive/KU, MAT-OEK/Kandidat/Thesis/Thesis_linear_CDS/Gamma_Calibration/CDS_TS_plot.npz")
    t_mats_plots = data['t_mats_plots']
    survival=data['survival']
    Gamma = data['Gamma']
    default_prob = data['default_prob']
    gammas = data['gamma_hist']


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
    fig.savefig(os.path.join(save_path, f"Default_curve.png"), dpi=150)
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
            # I1, I2, prot = model.get_CDS_deterministic(t[t_idx],t[t_idx],mat_grid[t_mat],
            #                                                        gammas[t_idx,:],t_grid_payments,t_mats) 

            CDS_implied[t_idx,t_mat] = prot / (I1+I2)
        print(f'Done with {(t_idx+1)/ CDS_obs.shape[0]}')

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
    plt.savefig(os.path.join(save_path, f"Reconstructed Spreads.png"), dpi=150)
    plt.close(fig)


    test = 1

    test = 1