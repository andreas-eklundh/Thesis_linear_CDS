import numpy as np 
from scipy.optimize import minimize, NonlinearConstraint, Bounds 

from numba import njit, float64, int64 
#from Models.ATSMGeneral.ATSM import ATSM
from Models.BaselineCIR.UnscentedKalman import KalmanUnscented as UKalman
from Models.BaselineCIR.UnscentedKalman import KalmanUnscentedFit as UKalmanFit
from scipy.stats import norm, ncx2, gamma, expon

from numba.experimental import jitclass 

spec = [
    ('kappa', float64[:]),        # row/column -> flatten to 1D
    ('theta', float64[:]),
    ('sigma', float64[:]),    # row vector -> 1D
    ('kappa_p', float64[:]),        # row/column -> flatten to 1D
    ('theta_p', float64[:]),
    ('delta', float64), 
    ('tenor',  float64),
    ('r',float64)
    ]



@jitclass(spec)
class CIR_Theta:
    def __init__(self, kappa,theta,sigma,kappa_p,theta_p, delta, tenor,r, ):
        self.kappa = np.ascontiguousarray(kappa)
        self.theta = np.ascontiguousarray(theta)
        self.sigma = np.ascontiguousarray(sigma)

        # For computing Kalman Transition and measurement (but h is CDS, so must be computed as pricing)
        self.kappa_p = np.ascontiguousarray(kappa_p)
        self.theta_p = np.ascontiguousarray(theta_p)

        self.delta = delta
        self.tenor = tenor
        self.r  = r




# For reference, also the solution in Lando (2004).
# Potentially more handystable (numba). Default is rho=1 as it will be the one we use (maybe?)
@njit
def cir_solution(cir_n,x,T,rho=1):
    # Local copies of kappa, theta to minimize code. Rename to comply with Lando.
    kappa,mu,sigma1 = cir_n.kappa, cir_n.theta, cir_n.sigma
    
    gamma = np.sqrt(kappa**2 + 2*sigma1**2*rho)

    beta_nom = (- 2 * rho * (np.exp(gamma*T)-1) + 
                    x * np.exp(gamma * T) * (gamma - kappa) + 
                    x * (gamma + kappa))
    
    beta_denom = (2 * gamma + 
                    (gamma + kappa - x * sigma1**2) * (np.exp(gamma * T) - 1))

    beta = beta_nom / beta_denom

    alpha_log_nom = (2 * gamma * np.exp((gamma + kappa ) * T / 2))

    alpha_log_denom = (2 * gamma + 
                        (gamma + kappa - x * sigma1**2)*(np.exp(gamma * T)-1))
    alpha = (2 * kappa * mu * 
                np.log(alpha_log_nom/alpha_log_denom)
                / sigma1**2) 
    
    return alpha,beta
## Generate generic functions to calculate expected values etc.


# Derivatives of alpha, beta. 
@njit
def cir_derivatives(cir_n,x,T,rho=1):
    kappa,mu,sigma1 = cir_n.kappa, cir_n.theta, cir_n.sigma
    gamma = np.sqrt(kappa**2 + 2*sigma1**2*rho)

    denom = (2 * gamma + (gamma + kappa - x * sigma1**2)*(np.exp(gamma * T)-1))

    # Beta
    bterm1  = - (2*rho*(np.exp(gamma*T)-1)**2 * sigma1**2) / denom**2
    bterm2 = (np.exp(gamma * T)*(gamma-kappa) + (gamma+kappa)) / denom
    # This  is tge second term of derivatives from the 'bottom'
    bterm3 = x * (np.exp(gamma * T)*(gamma-kappa) + (gamma+kappa)) *sigma1**2 *(np.exp(gamma*T)-1) / denom**2

    beta_x = bterm1 + bterm2 + bterm3

    # Alpha. 
    #alpha_x = 2 * kappa * mu * (2 * gamma + (gamma + kappa -x * sigma1**2) * (np.exp(gamma * T)-1))*(np.exp(gamma * T) - 1)
    alpha_x = 2 * kappa * mu *(np.exp(gamma * T) - 1) / (2 * gamma + (gamma + kappa -x * sigma1**2) * (np.exp(gamma * T)-1))

    return alpha_x, beta_x

# The Laplace Transform
@njit
def Laplace_Transform(cir_n,lambda_t, x, T):
    alpha,beta = cir_solution(cir_n,x,T,rho=1)

    # Return value of Laplace Transform - Specific vals of w->ZCB price.
    
    return np.exp(alpha + beta @ lambda_t)[0]



# Might need som logarithms on the lambda end. 

# Coupon leg 'easy' should be similar to a ATSM 
@njit
def calc_coupon_leg(cir_n,t,t_mat, lambda_t):
    I = np.zeros(1)
    t_grid_len = int(np.round(np.round(t_mat - t) / cir_n.tenor)) + 1
    t_grid = np.zeros(t_grid_len+1)
    for i in range(t_grid_len+1):
        t_grid[i] = t + i * cir_n.tenor
    x = np.array([0]) # Look iinto plug straight into. Also look into grid comp.

    for t_idx in range(1, len(t_grid)):
        expectation = Laplace_Transform(cir_n,lambda_t, x, t_grid[t_idx] - t)
        I += (t_grid[t_idx]-t_grid[t_idx-1]) * np.exp(-cir_n.r * (t_grid[t_idx] - t)) * expectation

    return I


# Accrued leg. Think it is going to follow similar to protection leg. 
# so (46 on 23/59) with the additional increment term.
# Helper function to get the grid.
# TODO_ Look into this.
@njit
def _get_default_grid(u, t_grid):
    for idx in range(len(t_grid) - 1):
        if (u > t_grid[idx]) & (u <= t_grid[idx + 1]):
            return (u - t_grid[idx])

    return 0.0 # Or some other appropriate default value

@njit
def accrual_integrand(u, cir_n,t,t_mat,lambda_t):
    t_grid_len = int(np.round(np.round(t_mat - t) / cir_n.tenor)) + 1
    t_grid = np.zeros(t_grid_len+1)
    for i in range(t_grid_len+1):
        t_grid[i] = t + i * cir_n.tenor
    integrand = np.real(
    (np.exp(-cir_n.r * (u-t)) * _get_default_grid(u,t_grid) *  
        (cir_derivatives(cir_n,0,u-t)[0] + 
        cir_derivatives(cir_n,0,u-t)[1] @ lambda_t) *
        Laplace_Transform(cir_n,lambda_t, 0, u - t)
    )
    )
    return integrand

@njit
def calc_accrual_leg(cir_n, t,t_mat, lambda_t):
    Ai_val = trapezoidal_rule(accrual_integrand,t,t_mat,cir_n, t,t_mat, lambda_t)
    return Ai_val

# Protection leg:

@njit
def protection_integrand(u, cir_n, t, lambda_t):
    return np.real(
        ((1- cir_n.delta)*np.exp(-cir_n.r * (u-t)) * (cir_derivatives(cir_n,0,u-t)[0] + 
          cir_derivatives(cir_n,0,u-t)[1] @ lambda_t)*
         Laplace_Transform(cir_n,lambda_t, 0, u - t)
        )
    )

@njit
def calc_protection_leg(cir_n,t,t_mat, lambda_t):
    prot_val = trapezoidal_rule(protection_integrand,t,t_mat,cir_n, t, lambda_t)

    return prot_val



# This is the h in the unscented Kalman Filter. L 
@njit
def calc_CDS(cir_n,t,t_mat, lambda_t):
    prot_val = calc_protection_leg(cir_n,t,t_mat, lambda_t)
    I1 = calc_coupon_leg(cir_n,t,t_mat, lambda_t)
    I2 = calc_accrual_leg(cir_n,t,t_mat, lambda_t)

    return prot_val / (I1 + I2 )


def calc_CDS_deriv(cir_n,t,t_mat, lambda_t):
    prot_val = calc_protection_leg(cir_n,t,t_mat, lambda_t)
    I1 = calc_coupon_leg(cir_n,t,t_mat, lambda_t)
    I2 = calc_accrual_leg(cir_n,t,t_mat, lambda_t)

    return prot_val / (I1 + I2)

# TODO: Check this code.
@njit
def trapezoidal_rule(integrand, a, b,*args):
    n_steps=50 # Maybe bup up if too little precision - may also increase optimizer precision...
    if n_steps <= 0:
        return 0.0

    delta = (b - a) / n_steps
    
    # Check for empty args to avoid issues
    # Pass a and b, then the *args to the integrand
    integral_sum = 0.5 * (integrand(a, *args) + integrand(b, *args))
    
    for i in range(1, n_steps):
        x = a + i * delta
        integral_sum += integrand(x, *args)
        
    return integral_sum * delta

# Assume calc_CDS is a function that can also be jitted.
# It should be defined and decorated with @njit.

# Refactor the make_h function to accept cir_n as a direct argument.

@njit
def cds_spread(cir_n, X, t, t_mat_grid):
    result = np.zeros(t_mat_grid.shape[0], dtype=np.float64)
    for i in range(t_mat_grid.shape[0]):
        # Pass a scalar from the array X
        result[i] = calc_CDS(cir_n, t, t_mat_grid[i], X)[0] # A
    
    return result

# Linear approx for extended Kalman.
# def cds_spread(cir_n, X, t, t_mat_grid):
    
#     result = np.zeros(t_mat_grid.shape[0], dtype=np.float64)
#     for i in range(t_mat_grid.shape[0]):
#         # Pass a scalar from the array X
#         result[i] = calc_CDS(cir_n, t, t_mat_grid[i], X)[0] # A
        
#     return result


## Class to actually call
class CIRIntensity():
    def __init__(self, r, delta, tenor):
        self.r = r
        self.delta = delta
        self.tenor = tenor

        # Set parameters randomly at first, but so something is present.
        self.set_params(params = None)

    def set_params(self,params):
        # If no parameters are set, use random. 
        # Also for simulatio    n purposes
        if params is None:
            rng = np.random.default_rng()  # independent each time
            self.kappa = rng.uniform(0.1, 1, size=(1,))
            self.theta =  rng.uniform(0.1, 0.7, size=(1,))
            self.sigma =  rng.uniform(0.01, 0.1, size=(1,))
            # initialise all positive. 

            self.kappa_p = self.kappa
            self.theta_p = self.theta
            self.sigma_err = np.random.uniform(0.001, 0.0099, size=(1,))


        else:
            # Else, asusming some paramter tuning, set then here.
            self.kappa, self.theta, self.sigma,self.kappa_p, self.theta_p,self.sigma_err = params


        # Build 'matrices for initialization of ATM class. 
        # K0 = np.array([self.kappa * self.theta])
        # K1 = - np.array([self.kappa]).reshape((1,1))
        # H0 = np.array([0]).reshape((1,1))
        # # Think about this shape.
        # H1 = np.ones(shape=(1,1,1)) * self.sigma**2
        # Also check up on this
        # rho0,rho1 = np.array([0]), np.array([1])

        # Create AffimeTSM to carry around.
        # NOTE: NOT NECESSARILY 100% APPLICABLE. But solve RICATTI EQ should be doable.
        # Note below is necessary in broader class.
        # self.atsm = ATSM(K0,K1,H0,H1, rho0,rho1)


    # For testing later...
    # Simulation will likely also be th eway to go about expression in Filipovic (tedious)
    def simulate_intensity(self, lambda0,T,M,scheme, measure):
        if measure == 'P':
            theta = self.theta_p
            kappa = self.kappa_p
        elif measure == 'Q':
            theta = self.theta
            kappa = self.kappa
        # Do baseline calculations
        delta = T / M
        T_return = np.array([0.00001] + [delta*k for k in range(1,M+1)])
        path = np.full(shape = M + 1, fill_value=lambda0) # to include zero.
        W = norm.rvs(size = M) # simulate at beginning - faster!
        if scheme == "Euler":
            for i in range(1,M+1):
                mu_t = kappa*(theta - path[i - 1])
                sigma_t = self.sigma * np.sqrt(path[i-1])
                path[i] = path[i - 1] + delta*mu_t + sigma_t * np.sqrt(delta) * W[i-1]
        elif scheme == "Milstein":
            for i in range(1,M+1):
                mu_t = kappa*(theta - path[i - 1])
                sigma_t = self.sigma * np.sqrt(path[i-1])
                sigma_prime_t = self.sigma * 1 / (2*np.sqrt(path[i-1]))
                path[i] = path[i - 1] + delta*mu_t + sigma_t * np.sqrt(delta) * W[i-1] + 1/2 * sigma_prime_t * sigma_t * delta*(W[i-1]**2-1)
        elif scheme == "Exact":
            for i in range(1,M+1):
                k = 4 * kappa * theta / (self.sigma**2)
                l = 4 * kappa * np.exp(-kappa * T_return[i]) / (self.sigma**2 * (1-np.exp(-kappa *T_return[i]))) * path[i-1]
                factor = self.sigma**2 * (1 - np.exp(-kappa * T_return[i])) / (4*kappa)
                path[i] = factor * ncx2.rvs(df = k, nc = l)

        return T_return, path
    
    # Since we are going to simulate, we will also need to simulate defaults.
    def get_cdso_pric_MC(self,t,t0,t_M,strike,lambda0,N,M):
        cir_n = CIR_Theta(self.kappa,self.theta,self.sigma,
                       self.kappa_p,self.theta_p,
                       self.delta,self.tenor,self.r)
        # N prices are comuted and averaged MC
        prices = np.zeros(shape = N)
        for i in range(N):
            # Get default intensity process. 
            T_return,lambda_t = self.simulate_intensity(lambda0,t0[0],M,scheme = 'Milstein')
            # Compute prob of default at time t0
            deltas = np.array([T_return[i]-T_return[i-1] for i in range(1,M+1)])
            Lambda = np.cumsum(lambda_t[1:]*deltas)
            # survival_process = np.exp(-np.cumsum(lambda_t[1:]*deltas)) # only approximates
            # default_process = 1-survival_process
            # Determine if default or not at t0. If lambda>E\simEXPo(1) option payoff is zero.
            E = expon.rvs()
            if Lambda[-1] >= E:
                prices[i] = 0
            # Else - begin to compute prices. 
            else: 
                lambda_end = np.array([lambda_t[-1]])
                prot = calc_protection_leg(cir_n, t0, t_M, lambda_end)
                # Quick fix due to way its written
                I1 = calc_coupon_leg(cir_n, t0[0], t_M[0], lambda_end)
                I2 = calc_accrual_leg(cir_n, t0[0], t_M[0], lambda_end)

                Value_CDS = prot - strike * (I1 + I2)

                # Discount back: 
                # Note still an option, so only enter if positive. 

                prices[i] = np.exp(-self.r * (t0 - t)) * np.maximum(Value_CDS,0)

        price_MC = np.mean(prices)

        return price_MC
    
    
    def Kalman_func(self, params, t_obs, t_mat_grid, CDS_obs):
        print(params)
        self.set_params(params)
        cir_params = CIR_Theta(self.kappa,self.theta,self.sigma,
                               self.kappa_p,self.theta_p,self.delta,
                               self.tenor,self.r)

        return UKalmanFit(params,cir_params,t_obs,t_mat_grid,CDS_obs,cds_spread)


    def Kalman_out(self, params, t_obs, t_mat_grid, CDS_obs):
        self.set_params(params)
        cir_params = CIR_Theta(self.kappa,self.theta,self.sigma,
                               self.kappa_p,self.theta_p,self.delta,
                               self.tenor,self.r)

        # print(params)
        return UKalman(params, cir_params,t_obs,t_mat_grid,CDS_obs,cds_spread)



    def penalized_obj(self,params, t_obs, t_mat_grid, CDS_obs, base_func):
        """
        Penalized objective for Nelder-Mead.
        base_func = your Kalman_func
        """
        # print(params)
        # Unpack
        kappa, theta, sigma, kappa_p, theta_p, sigma_err = params

        # positivity constraints (soft bounds)
        if np.any(params <= 0): 
            return 1e12

        # Feller condition: 2*kappa_p*theta_p - sigma^2 >= 0
        feller_val = 2 * kappa_p * theta_p - sigma**2
        if feller_val < 0:
            return 1e12

        # Add feller under both
        feller_val = 2 * kappa * theta - sigma**2
        if feller_val < 0:
            return 1e12

        else:
        # Base objective (e.g. Kalman filter log-likelihood)
            base_val = base_func(params, t_obs, t_mat_grid, CDS_obs)

            return base_val


    ### ALL THAT KALMAN STUFF. Seems right, likely logal mininima.
    def kalmanfilter(self,t_obs, t_mat_grid, CDS_obs):
        # Construct matrices for fitting model.
        x0 = np.array([self.kappa,self.theta,self.sigma,self.kappa_p,self.theta_p, self.sigma_err]).flatten()

        # res = minimize(self.Kalman_func, x0,
        #                args=(t_obs, t_mat_grid, CDS_obs), method='Nelder-Mead')

        # # If too small parameters, might get stuck at zero/undefined values.
        # bounds = Bounds(
        #     [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6],  # lower bounds
        #     [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]  # upper bounds
        # )
        # feller_con_p = NonlinearConstraint(lambda p: 2*p[0]*p[1] - p[2]**2, 0, np.inf)
        # feller_con_q = NonlinearConstraint(lambda p: 2*p[3]*p[4] - p[2]**2, 0, np.inf)
        # res = minimize(
        #     self.Kalman_func,   # you don’t need penalized_obj anymore!
        #     x0,
        #     args=(t_obs, t_mat_grid, CDS_obs),
        #     method="trust-constr",
        #     bounds=bounds,
        #     constraints=[feller_con_p, feller_con_q],
        #     options={
        #         "xtol": 1e-10,         # parameter tolerance
        #         "gtol": 1e-10,         # gradient tolerance
        #         "barrier_tol": 1e-12,  # feasibility tolerance for constraints
        #         "maxiter": 5000,
        #         "verbose": 3
        #     }
        # )

        res = minimize(
            self.penalized_obj,
            x0,
            args=(t_obs, t_mat_grid, CDS_obs, self.Kalman_func),
            method='Nelder-Mead',
            options = {
                "xatol": 1e-6,
                "fatol": 1e-6,
                "maxiter": 500,
                "disp": True
            }
        )  
        
        params_opt = res.x

        opt_likelihood = res.fun

        self.set_params(params_opt)

        print(params_opt, opt_likelihood)

        return params_opt,opt_likelihood

    
    def run_n_Kalman(self,t_obs, t_mat_grid, CDS_obs, n_optim=5):
    
        # Define grid of values. 
        current_objective = 1e10 #very high objective, hopefully inrealistic.
        for i in range(n_optim):
            print(f"Optimization {i+1}")
            self.set_params(params=None)
            params_opt,opt_likelihood = self.kalmanfilter(t_obs, t_mat_grid, CDS_obs)
            if opt_likelihood < current_objective:
                print(f"New optimal parameters at iteration {i+1}.")
                final_param = params_opt
                # B-ump current opjective.
                current_objective = opt_likelihood


        # Get values for done optimization. 
        Xn,Zn, Pn = self.Kalman_out(final_param, t_obs, t_mat_grid, CDS_obs)
        # Set parameters, so eimulation is possible.
        self.set_params(params=final_param)

        # Return optimal states.
        print("Kalman Filter Done") 
        return final_param, Xn,Zn, Pn
        
def __main__():
    cir = CIRIntensity(0.0252, 0.4, 0.25)
    x0 = np.array([0])
    #alpha_num, beta_num = cir.solve_ricatti(beta_0=x0,alpha_0=0,T=1)
    # alpha_lando,beta_lando = cir.cir_solution(x=0,T=1)

    # #print(f'Numerical alpha,beta: {alpha_num,beta_num}')
    # print(f'Lando alpha,beta: {alpha_lando,beta_lando}


    import pandas as pd
    t_grid = [0 + 0.25* i for i in range(0, int(10 / 0.25)+1)]
    # Test of grid

    # _get_default_grid(0.5,t_grid)    

    #### Preliminary investigation. 
    test_df = pd.read_excel("./Data/test_data.xlsx")

    # Pivot
    test_df = test_df.pivot(index = ['Date','Ticker'],
                            columns='Tenor',values = 'Par Spread').reset_index()
    # Test on subset data ownly to get very few obs. One large spread increase to test.
    #test_df = test_df[(test_df['Date']<'2021-01-01') & (test_df['Date']>='2019-06-01')]
    test_df = test_df[5::5]

    # Function to convert tenors to months to same metric (so
    test_df['Years'] = ((test_df['Date'] - test_df['Date'].min()).dt.total_seconds() / (365.25 * 24 * 3600)).drop_duplicates()
    t = np.array(test_df['Years'])
    # These are all available time points. 


    # mat_grid = np.array([1,2,3,4,5,7,10,15,20,30])
    mat_grid = np.array([5]) # Matures in 5 years, but specific dates.
    t_mat_grid = np.ascontiguousarray(mat_grid[:, None] + t[None, :])   # shape (len(T_M_grid), len(t_obs))

    # CDS_obs = np.array(test_df[['1Y','2Y','3Y','4Y','5Y','7Y','10Y','15Y',
    #                    '20Y','30Y']])
    CDS_obs = np.array(test_df[['5Y']] ) # ,'5Y','10Y']]) 
    cir_params = CIR_Theta(cir.kappa,cir.theta,cir.sigma,
                        cir.kappa_p,cir.theta_p,
                        cir.delta,cir.tenor,cir.r)

    # final_param = np.array([cir.kappa,cir.theta,cir.sigma, cir.kappa_p,cir.theta_p, np.array([0.03])]).flatten()
    # # Fit the model
    # Xn,Zn, Pn = cir.Kalman_out(final_param, t, t_mat_grid, CDS_obs)

    final_param, Xn,Zn, Pn = cir.run_n_Kalman(t,t_mat_grid,CDS_obs,n_optim=6)

    # # Save parameters somewhere for callable again. 
    np.savez("C:/Users/andre/OneDrive/KU, MAT-OEK/Kandidat/Thesis/Thesis_linear_CDS/Results/Kalman_resultsCIR.npz",
            final_param=final_param,
            Xn=Xn,
            Zn=Zn,
            Pn = Pn)

    data = np.load("C:/Users/andre/OneDrive/KU, MAT-OEK/Kandidat/Thesis/Thesis_linear_CDS/Results/Kalman_resultsCIR.npz")
    final_param = data["final_param"]
    Xn = data["Xn"]
    Zn = data["Zn"]
    Pn = data["Pn"]



    import matplotlib.pyplot as plt
    import os
    ### Test: Compare To CDS. 
    save_path = f"./Exploratory/"   # <--- change to your path
    os.makedirs(save_path, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(t, Xn.flatten(), label=f"Default intensity")
    #ax.plot(t, pred_Xn.flatten(), label=f"Predictions")

    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Default Intensity")
    ax.set_title("Default itensity CIR model, Kalman estimation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, "KalmanCIR.png"), dpi=150)
    plt.close(fig)


    ### Test: Compare To CDS.   
    fig, ax = plt.subplots(figsize=(10,6))
    for i in range(Zn.shape[1]):
        #ax.plot(t, pred_Zn[i,:].flatten(), label=f"Predicted CDS Spread, Maturity {mat_grid[i]}")
        ax.plot(t, CDS_obs[:, i], "o", alpha=0.7, label=f"Observed {mat_grid[i]}Y CDS")
        ax.plot(t, Zn[:,i].flatten(), "-", linewidth=2, label=f"Kalman Filter {mat_grid[i]}Y CDS")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("CDS Spread")
    ax.set_title("Kalman CDS Spreads")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, "CDSKalman_Check.png"), dpi=150)
    plt.close(fig)


    #### Option pricing using paramters. Would be a MC approach likely. 
    # Might be possible with a Fourier approach? 

    # Simulate a few paths to test.

    # Simulate default intensities starting from 'today' (either today is start of period or last day)
    # Last day makes the most sense, as we then estimate prices going forward. 

    T_simul = 10
    M = 50
    delta = T_simul / M
    T_plot = np.array([i*delta for i in range(0,M+1)])
    np.random.seed(11)
    T_return, lambda_euler = cir.simulate_intensity(lambda0=Xn[-1,:],T=T_simul,M=M,scheme="Euler")
    np.random.seed(11)
    T_return,lambda_mil = cir.simulate_intensity(lambda0=Xn[-1,:],T=T_simul,M=M,scheme="Milstein")
    np.random.seed(11)
    T_return,lambda_ex = cir.simulate_intensity(lambda0=Xn[-1,:],T=T_simul,M=M,scheme="Exact")


    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(T_plot, lambda_euler, label=f"Euler scheme,M={M}")
    ax.plot(T_plot, lambda_mil, label=f"Milstein scheme,M={M}")
    #ax.plot(T_plot, lambda_ex, label=f"Exact scheme,M={M}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Intensity Process")
    ax.grid()
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, "simul_cir.png"), dpi=150)
    plt.close(fig)



    ### Attempt to calculate CDSO prices - 1Y5Y year credit default swaption.
    t = np.array([0])
    t_start = np.array([1])
    T_option = t_start + 5

    # Get time zero stats. 
    strike_spreads = np.array([250,300,350]) / 10000
    priceCDSO = np.zeros(strike_spreads.shape[0])


    for i,k in enumerate(strike_spreads):
        priceCDSO[i] = cir.get_cdso_pric_MC(t=t,t0=t_start,
                                    t_M = T_option,strike = k,
                                    lambda0 = Xn[-1,:],N = 1000,M = 10000)


    print(f'CDSO prices {priceCDSO} for spreads in {strike_spreads}')


    test = 1

    test = 1