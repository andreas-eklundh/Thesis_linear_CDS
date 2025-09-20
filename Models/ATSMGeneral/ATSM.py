from scipy.integrate import solve_ivp
import numpy as np
from scipy.stats import multivariate_normal as mnorm
from scipy.integrate import quad

### Laplace Transform for Affine models and option pricing. 
# Q1: How general can these solutions be programmed?
# Q2: Should there be a discounted version or not (probably just a special case)

class ATSM():
    def __init__(self,K0,K1,H0,H1, rho0,rho1, l0=None, l1=None, jump_dens = mnorm.pdf ):
        # Define essential vectors, matrices etc to take further on.
        self.K0 = K0
        self.K1 = K1
        self.H0 = H0
        self.H1 = H1
        self.rho0 = rho0
        self.rho1 = rho1

        # If not specified, make sure jumpes are zero.
        if l0 is None:
            self.l0 = 0
        else:
            self.l0 = l0

        if l1 is None:
            self.l1 = np.zeros(self.K0.shape[0])
        else:
            self.l1 = l1
        
        self.density = jump_dens
    # To account for Jumps, we need to calculate Laplace of counting process
    # One would need some very general implementation.
    # NEED: Distribution of Jump sizes
        
    def theta(self, omega):
        def univariate_cf(wj):
            # Dynamically truncate bounds to avoid overflow from large Im(wj) * x
            bound = min(100, 30 / (np.abs(np.imag(wj)) + 1e-6))  # prevent div by 0
            integrand = lambda x: np.exp(1j * wj * x) * self.density(x)
            result, _ = quad(integrand, -bound, bound, limit=200, epsabs=1e-10, epsrel=1e-10)
            return result

        cf_values = [univariate_cf(wj) for wj in omega]
        return np.prod(cf_values)

    # Function for computing the odd tensor product. 
    def _tensor_prod(self, beta): 
        return np.einsum('i,ijk,j->k', beta, self.H1, beta)


    def solve_ODE_system(self, beta_0,alpha_0, T):
        init_conds = np.append(beta_0,alpha_0)
        def primes(t, betaalpha):
            beta = betaalpha[:beta_0.shape[0]]
            alpha = betaalpha[-1] 
            # Case wise if there are jumps or not (slow)
            if ((self.l0 == 0) & (np.sum(self.l1) == 0)):
                 dbeta = self.K1.T @ beta + 0.5 * self._tensor_prod(beta) - self.rho1 
                 dalpha =  self.K0.T @ beta + 0.5 *  beta.T @ self.H0 @ beta  - self.rho0
            else: 
                dbeta = self.K1.T @ beta + 0.5 * self._tensor_prod(beta) + self.l1 * (self.theta(beta)-1) - self.rho1 
                dalpha =  np.matmul(self.K0.T, beta) + 0.5 *  beta.T @ self.H0 @ beta + self.l0 * (self.theta(beta)-1) - self.rho0

            try:
                return np.concatenate([dbeta, [dalpha]])
            except:
                return np.concatenate([dbeta, dalpha])


        # Note revert T,0 to go backwoards in time?
        ode_sol =  solve_ivp(fun=primes,  t_span=[0,T], y0=init_conds , t_eval=[T],
                         method='RK45', rtol=1e-15, atol=1e-15)
        self.beta, self.alpha =  ode_sol.y[:beta_0.shape[0]].flatten(), ode_sol.y[-1].item()


    # The Laplace Transform
    def Laplace_Transform(self, X, w, T):
        self.solve_ODE_system(w,0 + 1j*0, T) # Incorporated initial conditions


        # Return value of Laplace Transform - Specific vals of w->ZCB price.
        
        return np.exp(self.alpha + self.beta @ X)
    

    def G_transform(self,y,a,b,Xt,T):
        first_term = self.Laplace_Transform(Xt, a, T)
        Laplace_fixed = lambda v: self.Laplace_Transform(Xt,a + 1j * v * b, T)
        lower_G = 1e-6  # small shift away from 0
        upper_G = 100
        Laplace_int, _ = quad(
            lambda v: np.imag(Laplace_fixed(v) * np.exp(-1j * v * y)) / v,
            lower_G, upper_G,
            limit=200,
            epsabs=1e-12,
            epsrel=1e-12
        )

        return np.real(first_term/2 - Laplace_int/np.pi)



