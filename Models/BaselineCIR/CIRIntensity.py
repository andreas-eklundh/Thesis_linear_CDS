import numpy as np 
from scipy.optimize import minimize, NonlinearConstraint 
from numba import njit, float64, int64 
#from Models.ATSMGeneral.ATSM import ATSM
from UnscentedKalman import KalmanUnscented as UKalman
from UnscentedKalman import KalmanUnscentedFit as UKalmanFit

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
    
    return np.exp(alpha + beta @ lambda_t)



# Might need som logarithms on the lambda end. 

# Coupon leg 'easy' should be similar to a ATSM 
@njit
def calc_coupon_leg(cir_n,t,t_mat, lambda_t):
    I = np.zeros(1)
    t_grid_len = int(np.round(np.round(t_mat - t) / cir_n.tenor)) + 1
    t_grid = np.zeros(t_grid_len+1)
    for i in range(t_grid_len+1):
        t_grid[i] = t + i * cir_n.tenor
    x = np.array([0])

    for t_idx in range(1, len(t_grid)):
        expectation = Laplace_Transform(cir_n,lambda_t, x, t_grid[t_idx] - t)
        I += cir_n.tenor * np.exp(-cir_n.r * (t_grid[t_idx] - t)) * expectation

    return I


# Accrued leg. Think it is going to follow similar to protection leg. 
# so (46 on 23/59) with the additional increment term.
# Helper function to get the grid.
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
        cir_derivatives(cir_n,0,u-t)[1] * lambda_t) @
        Laplace_Transform(cir_n,lambda_t, 0, u - t)
    )
    )
    return integrand

@njit
def calc_accrual_leg(cir_n, t,t_mat, lambda_t):
    Ai_val = trapezoidal_rule(accrual_integrand,t,t_mat,cir_n, t,t_mat, lambda_t)
    return np.array([Ai_val])

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
    n_steps=100 # Maybe bup up if too little precision.
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

        else:
            # Else, asusming some paramter tuning, set then here.
            self.kappa, self.theta, self.sigma,self.kappa_p, self.theta_p = params[:5]


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
    def simulate_intensity(self, init_val):
        
        return None
    
    def get_cdso_price(t0,t,t_M,strike):
        return None
        
    def transform_params(self,p):
        kappa = np.exp(p[0])   # strictly > 0
        theta = np.exp(p[1])   # strictly > 0
        sigma = np.sqrt(2 * kappa * theta)  # ensures sigma^2 < 2*kappa*theta
        sigma_err = np.exp(p[3])  # strictly > 0
        return np.array([kappa, theta, sigma, sigma_err])
    
    def Kalman_func(self, params, t_obs, t_mat_grid, CDS_obs):
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
        n_mat = CDS_obs.shape[1]
        sigma_err = np.random.uniform(0.01, 0.099, size=(1,))

        x0 = np.array([self.kappa,self.theta,self.sigma,self.kappa_p,self.theta_p, sigma_err]).flatten()

        # res = minimize(self.Kalman_func, x0,
        #                args=(t_obs, t_mat_grid, CDS_obs), method='Nelder-Mead')

        # If too small parameters, might get stuck at zero/undefined values.
        # bounds = [
        # (1e-6, None),  # kappa > 0
        # (1e-6, None),  # theta > 0
        # (1e-6, None),  # sigma > 0
        # (1e-6, None),  # kappa_p > 0
        # (1e-6, None),  # theta_p > 0
        # (1e-6, None)   # sigma_err > 0
        # ]

        # feller_con = NonlinearConstraint(lambda p: 2*p[3]*p[4] - p[2]**2, 0, np.inf)

        res = minimize(
            self.penalized_obj,
            x0,
            args=(t_obs, t_mat_grid, CDS_obs, self.Kalman_func),
            method='Nelder-Mead',
            options = {
                "xatol": 1e-4,
                "fatol": 1e-4,
                "maxiter": 500,
                "disp": True
            }
        )  
        
        params_opt = res.x

        opt_likelihood = res.fun

        self.set_params(params_opt)

        print(params_opt, opt_likelihood)

        return params_opt,opt_likelihood

        #Xn,Zn, pred_Xn, pred_Pn, pred_Zn = self.Kalman_out(params_opt, t_obs, t_mat_grid, CDS_obs)

        # Return optimal states.
        #print("Kalman Filter Done") 
        #return Xn, Zn, pred_Xn, pred_Pn, pred_Zn
    
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

        # Return optimal states.
        print("Kalman Filter Done") 
        return final_param, Xn,Zn, Pn
    
cir = CIRIntensity(0.0252, 0.4, 0.25)
x0 = np.array([0])
#alpha_num, beta_num = cir.solve_ricatti(beta_0=x0,alpha_0=0,T=1)
# alpha_lando,beta_lando = cir.cir_solution(x=0,T=1)

# #print(f'Numerical alpha,beta: {alpha_num,beta_num}')
# print(f'Lando alpha,beta: {alpha_lando,beta_lando}


import pandas as pd
t_grid = [0 + 0.25* i for i in range(0, int(10 / 0.25)+1)]
# Test of grid

_get_default_grid(0.5,t_grid)    

#### Preliminary investigation. 
test_df = pd.read_excel("./Data/test_data.xlsx")

# Pivot
test_df = test_df.pivot(index = ['Date','Ticker'],
                        columns='Tenor',values = 'Par Spread').reset_index()
# Test on subset data ownly to get very few obs. One large spread increase to test.
# test_df = test_df[(test_df['Date']<'2021-01-01') & (test_df['Date']>='2019-06-01')]

test_df = test_df[5::5]

# Function to convert tenors to months to same metric (so
test_df['Years']= ((test_df['Date'] - test_df['Date'].min()).dt.total_seconds() / (365.25 * 24 * 3600)).drop_duplicates()
t = np.array(test_df['Years'])
# These are all available time points. 


# mat_grid = np.array([1,2,3,4,5,7,10,15,20,30])
mat_grid = np.array([5,10]) # Matures in 5 years, but specific dates.
#t_mat_grid = np.ascontiguousarray(mat_grid[:, None] + t[None, :])   # shape (len(T_M_grid), len(t_obs))

# CDS_obs = np.array(test_df[['1Y','2Y','3Y','4Y','5Y','7Y','10Y','15Y',
#                    '20Y','30Y']])
CDS_obs = np.array(test_df[['5Y']] ) # ,'5Y','10Y']]) 
#cir.kalmanfilter(t,mat_grid,CDS_obs,K=10)
cir_params = CIR_Theta(cir.kappa,cir.theta,cir.sigma,
                       cir.kappa_p,cir.theta_p,
                       cir.delta,cir.tenor,cir.r)


# Construct matrices for fitting model.
# n_mat = CDS_obs.shape[1]
#sigma_err =np.array([0.03])
#params = np.array([cir.kappa,cir.theta,cir.sigma,cir.kappa_p,cir.theta_p, sigma_err]).flatten()
#Xn,Zn, Pn = cir.Kalman_out(params, t,mat_grid, CDS_obs)


# Xn, pred_Xn, pred_Pn, pred_Zn = cir.Kalman_out(x0,  sigma_err, t, mat_grid, CDS_obs,K=3)

# cir.set_params(params = np.array([0.17057755, 0.01870321, 0.12281393, 0.11802593, 0.06389869, 0.00059881]))

final_param, Xn,Zn, Pn = cir.run_n_Kalman(t,mat_grid,CDS_obs,n_optim=10)

# Save parameters somewhere for callable again. 
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
ax.plot(t, Xn.flatten(), label=f"Default indensity")
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
    ax.plot(t[1:], Zn[1:,i].flatten(), "-", linewidth=2, label=f"Kalman Filter {mat_grid[i]}Y CDS")
ax.set_xlabel("Time (years)")
ax.set_ylabel("CDS Spread")
ax.set_title("Kalman CDS Spreads")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(save_path, "CDSKalman_Check.png"), dpi=150)
plt.close(fig)




#### to load the parameters again. 
# Load later
# data = np.load("kalman_results.npz")
# final_param = data["final_param"]
# Xn = data["Xn"]
# Zn = data["Zn"]
# pred_Xn = data["pred_Xn"]
# pred_Pn = data["pred_Pn"]
# pred_Zn = data["pred_Zn"]


#### Option pricing using paramters. Would be a MC approach likely. 
# Might be possible with a Fourier approach? 


test = 1