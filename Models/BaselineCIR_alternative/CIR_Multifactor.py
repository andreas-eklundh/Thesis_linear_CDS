import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, Bounds

from numba import njit, float64, int64
from Models.ATSMGeneral.ATSM import ATSM
from scipy.stats import norm, ncx2, gamma, expon
from scipy.integrate import quad
from scipy.linalg import expm
from numba.experimental import jitclass


## Class to actually call
class CIRIntensity():
    def __init__(self, r, delta, tenor,X_dim=1):
        self.r = r
        self.delta = delta
        self.tenor = tenor
        self.X_dim = X_dim
        # Set parameters randomly at first, but so something is present.
        self.set_params(params = None)

    def set_params(self,params,seed=None):
        X_dim = self.X_dim
        # If no parameters are set, use random.
        # Also for sithetalatio    n purposes
        if params is None:
            if seed == None:
                rng = np.random.default_rng()  # independent each time
            else:
                rng = np.random.default_rng(seed)  # independent each time

            self.kappa = rng.uniform(0.02, 0.1, size=(X_dim,))
            self.theta =  rng.uniform(0.001, 0.01, size=(X_dim,))
            self.sigma =  rng.uniform(0.001, np.sqrt(2*self.kappa*self.theta), size=(X_dim,))
            # initialise all positive.
            self.kappa_p = self.kappa + 0.1 # just initialise these clsoe to each other
            self.theta_p = self.theta + 0.01 # just initialise these clsoe to each other
            self.sigma_err = np.random.uniform(0.001, 0.01, size=(1,))


        else:
            # Else, asusming some paramter tuning, set then here.
            self.kappa, self.theta, self.sigma,self.kappa_p, self.theta_p,self.sigma_err = self.unpack_params(params)

    def unpack_params(self,params):
        X_dim = self.X_dim
        kappa, theta, sigma = params[:X_dim],params[X_dim:2*X_dim],params[2*X_dim:3*X_dim]
        kappa_p, theta_p,sigma_err = params[3*X_dim:4*X_dim],params[4*X_dim:5*X_dim], np.array([params[-1]])
        return kappa,theta,sigma,kappa_p,theta_p,sigma_err
    #### Solve affine equations.

    # For reference, also the solution in Lando (2004).
    # Potentially more handystable (numba). Default is rho=1 as it will be the one we use (maybe?)

    def cir_solution(self,params,x0,T,rho=1,corr=False):
        # Local copies of kappa, theta to minimize code. Rename to comply with Lando.
        kappa,theta,sigma1,kappa_p,theta_p,sigma_err = self.unpack_params(params)
        gamma = np.sqrt(kappa**2 + 2*sigma1**2*rho)
        # If x0 is one dimensional (intensity), use Lando forthetalas
        
        if corr == False:
            if isinstance(T,float):
                T_size = 1
            else:
                T_size = T.shape[0]
            
            # In this case, rho needs to be a vector (rho is rho 1 in actuality)
            alpha, beta = np.zeros((T_size)),np.zeros((T_size,self.X_dim)) 
            for i in range(self.X_dim):
                beta_nom = (- 2 * rho * (np.exp(gamma[i]*T)-1) +
                                x0[i] * np.exp(gamma[i] * T) * (gamma[i] - kappa[i]) +
                                x0[i] * (gamma[i] + kappa[i]))

                beta_denom = (2 * gamma[i] +
                                (gamma[i] + kappa[i] - x0[i] * sigma1[i]**2) * (np.exp(gamma[i] * T) - 1))

                beta[:,i] = beta_nom / beta_denom

                alpha_log_nom = (2 * gamma[i] * np.exp((gamma[i] + kappa[i] ) * T / 2))

                alpha_log_denom = (2 * gamma[i] +
                                    (gamma[i] + kappa[i] - x0[i] * sigma1[i]**2)*(np.exp(gamma[i] * T)-1))
                alpha += (2 * kappa[i] * theta[i] *
                            np.log(alpha_log_nom/alpha_log_denom)
                            / sigma1[i]**2)
                
            # Yields a vector of solutions. 
            return alpha,beta


        # If correlation or dgeneral solutions:
        else:
            # Build matrices for Afine term structure models.
            K0 = np.identity(self.X_dim) * kappa @ theta
            K1 = - np.identity(self.X_dim) * kappa
            H0 = np.zeros((self.X_dim,self.X_dim ))
            H1 = np.zeros((self.X_dim,self.X_dim,self.X_dim))
            for i in range(self.X_dim):
                H1[i,i,i] = sigma1[i]**2
            # Just assume first factor is default intensity.
            rho1 = np.ones(self.X_dim)
            # rho1[0] = 1

            atsm = ATSM(K0,K1,H0,H1,rho0=0,rho1=rho1)
            atsm.solve_ODE_system(x0,0,T)

            return atsm.alpha,atsm.beta

    def cir_derivatives(self,params,x,T,rho=1, corr=False):
        # Can work in 
        kappa,theta,sigma1,kappa_p,theta_p,sigma_err = self.unpack_params(params)
        gamma = np.sqrt(kappa**2 + 2*sigma1**2*rho)
        if corr == False:
            if isinstance(T,float):
                T_size = 1
            else:
                T_size = T.shape[0]
            
            alpha_x, beta_x = np.zeros((T_size)),np.zeros((T_size,self.X_dim)) 
            for i in range(self.X_dim):
                # In this case, rho needs to be a vector (rho is rho 1 in actuality)
                denom = (2 * gamma[i] + (gamma[i] + kappa[i] - x[i] * sigma1[i]**2)*(np.exp(gamma[i] * T)-1))

                # Beta
                bterm1  = - (2*rho*(np.exp(gamma[i]*T)-1)**2 * sigma1[i]**2) / denom**2
                bterm2 = (np.exp(gamma[i] * T)*(gamma[i]-kappa[i]) + (gamma[i]+kappa[i])) / denom
                # This  is tge second term of derivatives from the 'bottom'
                bterm3 = x[i] * (np.exp(gamma[i] * T)*(gamma[i]-kappa[i]) + (gamma[i]+kappa[i])) *sigma1[i]**2 *(np.exp(gamma[i]*T)-1) / denom**2

                beta_x[:,i] = bterm1 + bterm2 + bterm3

                # Alpha. 
                alpha_x += 2 * kappa[i] * theta[i] *(np.exp(gamma[i] * T) - 1) / (2 * gamma[i] + (gamma[i] + kappa[i] -x[i] * sigma1[i]**2) * (np.exp(gamma[i] * T)-1))

        return alpha_x, beta_x


    # The Laplace Transform
    def Laplace_Transform(self,params,lambda_t, T):
        x0 = np.zeros(self.X_dim) # This is how it is supposed to be...
        alpha,beta = self.cir_solution(params,x0,T,rho=1)
        # Return value of Laplace Transform - Specific vals of w->ZCB price.
        return np.exp(alpha + beta @ lambda_t).flatten()

    ##### Section on all the pricing stuff.
    # Coupon leg 'easy' should be similar to a ATSM 
    def calc_coupon_leg(self,params,t,t_mat, lambda_t):
        I = np.zeros(1)
        t_grid_len = int(np.round((t_mat - t) / self.tenor).item()) + 1
        t_grid = np.zeros(t_grid_len)
        for i in range(t_grid_len):
            t_grid[i] = t + i * self.tenor
        for t_idx in range(1, len(t_grid)):
            expectation = self.Laplace_Transform(params, lambda_t.T, t_grid[t_idx] - t)
            I += (t_grid[t_idx]-t_grid[t_idx-1]) * np.exp(-self.r * (t_grid[t_idx] - t)) * expectation
        return I


    # Accrued leg. Think it is going to follow similar to protection leg. 
    # so (46 on 23/59) with the additional increment term.
    # Helper function to get the grid.
    def _get_default_grid(self,u, t_grid):
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

    def calc_accrual_leg(self,params,t,t_mat, lambda_t):
        x = np.zeros(self.X_dim)
        t_grid_len = int(np.round((t_mat - t) / self.tenor).item()) + 1
        t_grid = np.zeros(t_grid_len)
        for i in range(t_grid_len):
            t_grid[i] = t + i * self.tenor
        integrand = lambda u: (np.exp(-self.r * (u-t)) * self._get_default_grid(u,t_grid) *  
            (self.cir_derivatives(params,x,u-t)[0] + 
            self.cir_derivatives(params,x,u-t)[1] @ lambda_t.T) *
            self.Laplace_Transform(params,lambda_t.T, u - t)
        )

        Ai_val, _ = quad(integrand,t,t_mat,epsabs=1e-9, epsrel=1e-9)
        return Ai_val

    # Protection leg:
    def calc_protection_leg(self,params,t,t_mat, lambda_t):
        x = np.zeros(self.X_dim)

        integrand = lambda u: ((1- self.delta)*np.exp(-self.r * (u-t)) * (self.cir_derivatives(params,x,u-t)[0] + 
            self.cir_derivatives(params,x,u-t)[1] @ lambda_t.T)*
            self.Laplace_Transform(params,lambda_t.T, u - t)
            )
        prot_val, _ = quad(integrand,t,t_mat,epsabs=1e-9, epsrel=1e-9)

        return prot_val



    def calc_CDS(self,params,t,t_mat, lambda_t):
        prot_val = self.calc_protection_leg(params,t,t_mat, lambda_t)
        I1 = self.calc_coupon_leg(params,t,t_mat, lambda_t)
        I2 = self.calc_accrual_leg(params,t,t_mat, lambda_t)

        return prot_val / (I1 + I2 )



    def cds_spread(self, X,params, t, t_mat_grid):
        result = np.zeros(t_mat_grid.shape[0], dtype=np.float64)
        for i in range(t_mat_grid.shape[0]):
            # Pass a scalar from the array X
            # Make sure that t grid is of size 1 due to logic.
            t_mat = np.array([t_mat_grid[i]])
            result[i] = self.calc_CDS(params,t, t_mat, X)[0] # A
        return result

    def Update_step(self,X_pred, P_pred, A,a, R_k, Y):
        # Step 3: Mean prediction, covariance, Kalman Gain etc.
        theta_k = A @ X_pred + a

        # covariance
        S_k = A @ P_pred @ A.T + R_k
        try:
            S_k_inv = np.linalg.inv(S_k)
        except:
            # Pseudo inverse if not working.
            S_k_inv = np.linalg.pinv(S_k)

        # Step 4: Compute Kalman Gain, filtered mean state, covariance.
        K_k = P_pred @ A.T @ S_k_inv
        vn = (Y - theta_k)
        m_k = X_pred + K_k @ vn
        P_k = P_pred - K_k @ S_k @ K_k.T

        return theta_k, vn,S_k, m_k, P_k

    # Prediction step:
    def Prediction_step(self,Xn, Pn, C,d, Q_k):
        # CIR transition.
        # Step 3: Mean prediction and cov predition.
        m_k = (C @ Xn +  d).flatten()

        # covariance -
        # Should be zero at first, then fill non Y cols. Same as done in Q_k
        P_k = np.zeros(shape = Q_k.shape)
        P_k = C @ Pn @ C.T + Q_k

        return m_k, P_k


    def get_cond_var(self,A, Sigma_prod, Delta):
        # Is logic can work completeley, just consider  A'=-A. Then same form
        Lambda, E = np.linalg.eig(A) # Just eigenvalues as array, columns are eigenvectors.

        S_bar = np.linalg.inv(E) @ Sigma_prod @ np.linalg.inv(E).T

        dim = A.shape[0] # assume every matrix is of same size and shape.

        # Then V_Delta.
        V_Delta = np.zeros((dim,dim))
        V_Delta_inf = np.zeros((dim,dim))

        for i in range(0,dim):
            for j in range(0,dim):
                exp_factor = (1-np.exp(-(Lambda[i] + Lambda[j]) * Delta))
                V_Delta[i,j] = S_bar[i,j] * exp_factor / (Lambda[i] + Lambda[j])
                V_Delta_inf[i,j] = S_bar[i,j] / (Lambda[i] + Lambda[j])

        Var = E @ V_Delta @ E.T
        Var_inf = E @ V_Delta_inf @ E.T


        return Var, Var_inf

    # Kalman filtering.
    def Kalman(self,params,t_obs, t_mat_grid, Y, result = False):
        # print(params)
        kappa,theta,sigma,kappa_p,theta_p,sigma_err = self.unpack_params(params)
        # Stop optimization if bad initial params.
        # positivity constraints (soft bounds)
        if np.any(params <= 0):
            return 1e12

        # Feller condition: 2*kappa_p*theta_p - sigma^2 >= 0
        feller_val = 2 * kappa_p * theta_p - sigma**2
        if np.any(feller_val < 0):
            return 1e12

        # Add feller under both - will hold also for matrices
        feller_val = 2 * kappa * theta - sigma**2
        if np.any(feller_val < 0):
            return 1e12

        n_obs = t_mat_grid.shape[1]
        n_mat = t_mat_grid.shape[0]

        Sigma = (np.identity(n_mat) * sigma_err**2)

        # Long term mean for each of the processes.
        alpha = 2 * kappa_p* theta_p / sigma**2
        beta = 2 *kappa_p / sigma**2
        # CIR values.
        X0 = alpha / beta
        P0 = alpha / beta**2
        # CIR conditional Mean and Variance based on parameters.
        L = X0.shape[0]

        Xn = np.zeros((L))
        Zn = np.zeros((n_mat))
        Pn = np.zeros((L,L))
        if result == True:
            Xn_out = np.zeros((n_obs,L))
            Zn_out = np.zeros((n_obs,n_mat))
            Pn_out = np.zeros((n_obs,L,L))
        # We want to store all predictions.
        pred_Xn  = X0
        pred_Pn  = ( np.identity(self.X_dim) * P0).reshape((self.X_dim,self.X_dim))
        Delta = t_obs[1] - t_obs[0] # Only apprx for now. Move to loop maybe.

        log_likelihood = 0
        # Run algo.
        x0_zcb = np.zeros(self.X_dim)
        kappa_P_diag = np.identity(self.X_dim) * kappa_p

        # To speed up, solve ricatti equations. Will be time homogenous.
        # Solve Ricatti Equations. Might move inside loop later - MUCH FASTER OUT HERE, IF SAME DIST APPROX.
        # THIS WILL LIKELY DO.
        a,A =  self.cir_solution(params,x0 = x0_zcb,T = t_mat_grid[:,0]- t_obs[0])
        a,A = -a,-A
        for n in range(0,n_obs):
            # UPDATE STEP
            Zn, vn,S_k, Xn, Pn = self.Update_step(pred_Xn,pred_Pn,A,a,Sigma,Y[n,:])
            # punish hashly if Xn below zero (mainly i). 
            # if np.any(Xn < 0 ) & (result == False):
            #     # Zn, vn,S_k, Xn, Pn = self.Update_step(pred_Xn,pred_Pn,A,a,Sigma,Y[n,:])
            #     return 1e12
            if result == True:
                Xn_out[n,:] = Xn
                Zn_out[n,:] = A @ Xn + a
                Pn_out[n,:,:] = Pn

            # Update log likelihood.
            det_S = np.abs(np.linalg.det(S_k))
            try:
                S_inv = np.linalg.inv(S_k)
            except:
                S_inv = np.linalg.pinv(S_k)

            log_likelihood += - 0.5 * (S_k.shape[0] * np.log(2*np.pi) + np.log(det_S) +
                                        vn.T @ S_inv @ vn
            )

            # Create arrays based on obs.
            phi_0 = (np.identity(kappa_P_diag.shape[0])-expm(-kappa_P_diag * Delta)) @ theta_p
            phi_X = expm(-kappa_P_diag * Delta)
            # Use CIR Variance for this (not going to need it)
            if (n < n_obs - 1): # Not sensible to predict further.
                # Works for Uncorrelated and indep noise.
                Q_t = (sigma**2 * theta_p * (1-np.exp(-kappa_p * Delta))**2 / (2 * kappa_p) +
                        Xn * sigma**2 * (np.exp(-kappa_p * Delta) - np.exp(-2*kappa_p * Delta))/ kappa_p)
                Q_t = ( np.identity(self.X_dim) * Q_t).reshape((self.X_dim,self.X_dim))

                Delta = t_obs[n+1] - t_obs[n]
                pred_Xn, pred_Pn = self.Prediction_step(Xn,Pn,phi_X,phi_0,Q_t)

        if result == True:
            return Xn_out,Zn_out, Pn_out
        else:
            return -log_likelihood


    def feller_constraint(self,params):
        d = self.X_dim
        # unpack
        kappa     = params[0:d]
        theta     = params[d:2*d]
        sigma     = params[2*d:3*d]
        kappa_p   = params[3*d:4*d]
        theta_p   = params[4*d:5*d]
        sigma_err = params[-1]

        # latent CIR Feller: 2*kappa*theta - sigma^2 >= 0
        feller_latent = 2*kappa*theta - sigma**2  # vector length d

        # observation Feller: 2*kappa_p*theta_p - sigma^2 >=0
        feller_obs = 2*kappa_p*theta_p - sigma**2  # vector length d

        # concatenate both
        return np.concatenate([feller_latent, feller_obs])  # length 2*d

    # Optimizer function.
    def run_kalman_filter(self, t_obs,t_mat_grid,Y,seed=1000):
        self.set_params(params=None,seed=seed)

        # Get initial values. Z
        x0 = np.concatenate([self.kappa, self.theta, self.sigma, self.kappa_p, self.theta_p,self.sigma_err])
        x0 = x0.flatten()

        # Try a different optimizer than nelder mead
        # d = self.X_dim  # number of factors

        # n_params = 5*d + 1  # adjust to actual number of params
        # lower_bounds = 1e-8 * np.ones(n_params)  # small positive number to avoid zero
        # upper_bounds = np.full(n_params, np.inf)

        # bounds = Bounds(lower_bounds, upper_bounds)
        # nonlinear_constraint = NonlinearConstraint(self.feller_constraint, 0, np.inf)

        # res = minimize(
        #     fun=self.Kalman,
        #     x0=x0,
        #     method='trust-constr',
        #     args=(t_obs, t_mat_grid, Y, False),
        #     bounds=bounds,
        #     constraints=[nonlinear_constraint],
        #     tol=1e-9,  # or whatever tolerance you need
        #     options={
        #         'maxiter': 100
        #     }
        # )
        res = minimize(
            fun = self.Kalman,
            x0 = x0,
            method='Nelder-Mead',
            args = (t_obs, t_mat_grid, Y,False),
            options = {
                "xatol": 1e-6,
                "fatol": 1e-6,
                "maxiter": 1000,
                "disp": True
            }
        )
        params = res.x
        self.kalman_obj  = res.fun
        # Run and return solution
        #Xn,Zn,Pn = self.Kalman(params,t_obs, t_mat_grid, Y,True)

        return params #, Xn,Zn,Pn
    
    def run_n_kalman(self, t_obs,t_mat_grid,Y,base_seed=1000,n_restarts=5):
        # Define grid of values. 
        current_objective = 1e10 #very high objective.
        for i in range(n_restarts):
            print(f"Optimization {i+1}")
            # Test several random points. 
            optim_params = self.run_kalman_filter(t_obs,t_mat_grid, Y,seed=base_seed+i)
            # Test new constraints

            if (self.kalman_obj < current_objective):
                print(f"New optimal parameters at iteration {i+1}.")
                current_objective = self.kalman_obj
                out_params = optim_params
                # Assuming we do get anything for optimizing.
                Xn_out,Zn_out,Pn_out = self.Kalman(out_params,t_obs, t_mat_grid, Y,True)

        return out_params, Xn_out,Zn_out,Pn_out
            
    # Simulation will likely also be th eway to go about expression in Filipovic (tedious)
    def simulate_intensity(self, lambda0,T,M,scheme, measure):
        # TODO: Make possible to simulate in several dim
        if measure == 'P':
            theta = self.theta_p
            kappa = self.kappa_p
        elif measure == 'Q':
            theta = self.theta
            kappa = self.kappa
        # Do baseline calculations
        delta = T / M
        X_dim = lambda0.shape[0]
        T_return = np.array([0.00001] + [delta*k for k in range(1,M+1)])
        path = (np.ones(shape = (X_dim,M + 1))* lambda0.reshape((X_dim,1)) ).T # to include zero.
        W = norm.rvs(size = (X_dim*M)).reshape((M,X_dim)) # simulate at beginning - faster!
        # Creat Matrices
        kappa_mat = np.diag(kappa.flatten())
        theta_vec = theta
        sigma_mat = np.diag(self.sigma.flatten())
        if scheme == "Euler":
            for i in range(1,M+1):
                mu_t = kappa_mat @ (theta_vec - path[i - 1,:])
                sigma_t = sigma_mat *  np.sqrt(path[i-1,:])
                path[i,:] = path[i - 1,:] + delta*mu_t +  np.sqrt(delta) * sigma_t @ W[i-1,:]
        elif scheme == "Milstein":
            for i in range(1,M+1):
                mu_t = kappa_mat @ (theta_vec - path[i - 1,:])
                sigma_t = sigma_mat *  np.sqrt(path[i-1])
                #sigma_prime_t = 
                path[i,:] = path[i - 1,:] + delta*mu_t +  np.sqrt(delta) * sigma_t @ W[i-1,:]+ 1/2 * sigma_prime_t * sigma_t * delta*(W[i-1]**2-1)
        elif scheme == "Exact":
            # TODO: Correct this.
            for i in range(1,M+1):
                k = 4 * kappa * theta / (self.sigma**2)
                l = 4 * kappa * np.exp(-kappa * T_return[i]) / (self.sigma**2 * (1-np.exp(-kappa *T_return[i]))) * path[i-1]
                factor = self.sigma**2 * (1 - np.exp(-kappa * T_return[i])) / (4*kappa)
                path[i] = factor * ncx2.rvs(df = k, nc = l)

        return T_return, path
    


    # MC pricing. 
    def get_cdso_pric_MC(self,params,t,t0,t_M,strike,lambda0,N,M):
        # N prices are comuted and averaged MC
        prices = np.zeros(shape = N)
        for i in range(N):
            # Get default intensity process. 
            T_return,lambda_t = self.simulate_intensity(lambda0,t0,M,scheme = 'Milstein',measure='Q')
            # Compute prob of default at time t0
            deltas = np.array([T_return[i]-T_return[i-1] for i in range(1,M+1)])
            Lambda = np.cumsum(lambda_t[1:]*deltas)
            # Determine if default or not at t0. If lambda>E\simEXPo(1) option payoff is zero.
            E = expon.rvs()
            if Lambda[-1] >= E:
                prices[i] = 0
            # Else - begin to compute prices. 
            else: 
                lambda_end = np.array([lambda_t[-1]])
                prot = self.calc_protection_leg(params, t0, t_M, lambda_end)
                # Quick fix due to way its written
                I1 = self.calc_coupon_leg(params, t0, t_M, lambda_end)
                I2 = self.calc_accrual_leg(params, t0, t_M, lambda_end)

                Value_CDS = prot - strike * (I1 + I2)

                # Discount back: 
                # Note still an option, so only enter if positive. 

                prices[i] = np.exp(-self.r * (t0 - t)) * np.maximum(Value_CDS,0)

        price_MC = np.mean(prices)

        return price_MC



    # # Derivatives of alpha, beta.
    # def cir_derivatives(self,params,x0,T,rho=1):
    #     kappa,theta,sigma1,kappa_p,theta_p,sigma_err = self.unpack_params(params)
    #     gamma = np.sqrt(kappa**2 + 2*sigma1**2*rho)

    #     denom = (2 * gamma + (gamma + kappa - x0 * sigma1**2)*(np.exp(gamma * T)-1))

    #     # Beta
    #     bterm1  = - (2*rho*(np.exp(gamma*T)-1)**2 * sigma1**2) / denom**2
    #     bterm2 = (np.exp(gamma * T)*(gamma-kappa) + (gamma+kappa)) / denom
    #     # This  is tge second term of derivatives from the 'bottom'
    #     bterm3 = x0 * (np.exp(gamma * T)*(gamma-kappa) + (gamma+kappa)) *sigma1**2 *(np.exp(gamma*T)-1) / denom**2

    #     beta_x = bterm1 + bterm2 + bterm3

    #     # Alpha.
    #     #alpha_x = 2 * kappa * theta * (2 * gamma + (gamma + kappa -x * sigma1**2) * (np.exp(gamma * T)-1))*(np.exp(gamma * T) - 1)
    #     alpha_x = 2 * kappa * theta *(np.exp(gamma * T) - 1) / (2 * gamma + (gamma + kappa -x0 * sigma1**2) * (np.exp(gamma * T)-1))

    #     return alpha_x, beta_x

