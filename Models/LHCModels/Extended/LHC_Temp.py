import numpy as np 
from scipy.optimize import minimize, NonlinearConstraint 
from scipy.stats import norm 
from scipy.integrate import solve_ivp 
from scipy.linalg import expm 
from scipy.optimize import lsq_linear 
from numba import njit, float64, int64 
from numba.experimental import jitclass 
import cvxpy as cp
from scipy.integrate import quad

import copy

spec = [
    ('a', float64[:,:]),        # row/column -> flatten to 1D
    ('c', float64[:,:]),
    ('gamma', float64[:,:]),    # row vector -> 1D
    ('b', float64[:,:]),
    ('beta', float64[:,:]),
    ('A', float64[:,:]),
    ('A_star', float64[:,:]),
    ('A_star_inv', float64[:,:]),
    ('id_mat', float64[:,:]),
    ('r', float64),
    ('m', int64),
    ('Y_dim', int64),
    ('delta', float64),
    ('tenor', float64),
    ('sum_Z',float64[:]),
    ('sum_D',float64[:]),
    
]



@jitclass(spec)
class LHCStruct:
    def __init__(self, a, c, gamma, b, beta, A, A_star, A_star_inv, id_mat, r, m, Y_dim, delta, tenor):
        self.a = np.ascontiguousarray(a)
        self.c = np.ascontiguousarray(c)
        self.gamma = np.ascontiguousarray(gamma)
        self.b = np.ascontiguousarray(b)
        self.beta = np.ascontiguousarray(beta)
        self.A = np.ascontiguousarray(A)
        self.A_star = np.ascontiguousarray(A_star)
        self.A_star_inv = np.ascontiguousarray(A_star_inv)
        self.id_mat = np.ascontiguousarray(id_mat)
        self.r = r
        self.m = m
        self.Y_dim = Y_dim
        self.delta = delta
        self.tenor = tenor

# -------- Matrix exponential approx (safe for numba) -------- #
@njit
def mat_exp_approx(A, dt, tol=1e-10):
    n = A.shape[0]
    I = np.eye(n)
    Adt = A * dt

    mat_expo = I.copy()
    term = I.copy()
    
    # We use a fixed upper limit to prevent infinite loops in cases of non-convergence
    limit = 20

    for i in range(1, limit + 1):
        # Calculate the next term
        term = np.dot(term, Adt) / i
        
        # # Check for convergence using the Frobenius norm
        # if np.linalg.norm(term, ord='fro') < tol:
        #     break  # Exit the loop if the term's contribution is negligible

        # Add the new term to the running sum
        mat_expo += term

    return mat_expo

# Rebuild dynamics and return the LHCStruct
def rebuild_lhc_struct(kappa, theta, gamma1, r, Y_dim, delta, tenor):
    m = theta.shape[0]
    
    # b: shape (m, Y_dim)
    b = np.zeros((m, Y_dim))
    b[m-1, 0] = theta[-1] * kappa[-1]

    # beta: shape (m, m)
    beta = np.zeros((m, m))
    for i in range(m):
        beta[i, i] = -kappa[i]
        if i + 1 < m:
            beta[i, i+1] = kappa[i] * theta[i]

    # gamma: row vector, shape (1, m)
    gamma = np.zeros((1, m))
    gamma[0, 0] = - gamma1

    # c: shape (Y_dim, Y_dim)
    c = np.zeros((Y_dim, Y_dim))

    # A: shape (Y_dim + m, Y_dim + m)
    A = np.zeros((Y_dim + m, Y_dim + m))
    A[:Y_dim, :Y_dim] = c
    A[:Y_dim, Y_dim:] = gamma
    A[Y_dim:, :Y_dim] = b
    A[Y_dim:, Y_dim:] = beta

    # Identity, A_star, and inverse
    id_mat = np.eye(Y_dim + m)
    A_star = A - r * id_mat
    A_star_inv = np.linalg.inv(A_star) # pinv always exist, not unique

    # a vector (assume ones for simplicity)
    a = np.ones((Y_dim,1))

    # Build and return the struct
    lhc = LHCStruct(a=a, c=c, gamma=gamma, b=b, beta=beta,
                    A=A, A_star=A_star, A_star_inv=A_star_inv,
                    id_mat=id_mat, r=r, m=m, Y_dim=Y_dim,
                    delta=delta, tenor=tenor)
    return lhc



# -------- psi functions rewritten for numba -------- #
@njit
def psi_Z(lhc, t, t_M):
    dt = t_M - t
    mat_exp = mat_exp_approx(lhc.A, dt)
    a0 = np.zeros(lhc.Y_dim + lhc.m)
    a0[:lhc.Y_dim] = lhc.a.ravel()
    return np.exp(-lhc.r * dt) * (a0 @ mat_exp).ravel()

@njit
def psi_D(lhc, t, t_M):
    dt = t_M - t
    mat_exp = mat_exp_approx(lhc.A_star, dt)
    # build [c | gamma]
    c_gamma = np.zeros((lhc.Y_dim, lhc.Y_dim + lhc.m))
    c_gamma[:, :lhc.Y_dim] = lhc.c
    c_gamma[:, lhc.Y_dim:] = lhc.gamma
    tmp = mat_exp - lhc.id_mat
    a_row = lhc.a.ravel()          
    return -(a_row @ c_gamma @ (lhc.A_star_inv @ tmp)).ravel()

@njit
def psi_D_star(lhc, t, t_M):
    dt = t_M - t
    mat_exp = mat_exp_approx(lhc.A_star, dt)
    c_gamma = np.zeros((lhc.Y_dim, lhc.Y_dim + lhc.m))
    c_gamma[:, :lhc.Y_dim] = lhc.c
    c_gamma[:, lhc.Y_dim:] = lhc.gamma
    term1 = dt * (lhc.A_star_inv @ mat_exp)
    term2 = lhc.A_star_inv @ ((lhc.id_mat * t - lhc.A_star_inv) @ (mat_exp - lhc.id_mat))
    a_row = lhc.a.ravel()
    return -(a_row @ c_gamma @ (term1 + term2)).ravel()

@njit
def psi_prot(lhc, t, t0, t_M):
    return (1.0 - lhc.delta) * (psi_D(lhc, t, t_M) - psi_D(lhc, t, t0))

@njit
def psi_prem(lhc, t, t0, t_M):
    sum_Z = np.zeros(lhc.Y_dim + lhc.m)
    sum_D = np.zeros(lhc.Y_dim + lhc.m)
    t_grid_len = int(np.round((t_M - t) / lhc.tenor).item()) + 1
    t_grid = np.zeros(t_grid_len)
    for i in range(t_grid_len):
        t_grid[i] = t + i * lhc.tenor
    for j in range(1, t_grid_len):
        dt = t_grid[j] - t_grid[j-1]
        sum_Z += dt * psi_Z(lhc, t, t_grid[j])
        if j < t_grid_len - 1:
            sum_D += dt * psi_D(lhc, t, t_grid[j])
    return (sum_Z + psi_D_star(lhc, t, t_M) - psi_D_star(lhc, t, t0)
            + t_grid[-2] * psi_D(lhc, t, t_M) - sum_D - t0 * psi_D(lhc, t, t0))

@njit
def psi_cds(lhc, t, t0, t_M, k):
    return psi_prot(lhc, t, t0, t_M) - k * psi_prem(lhc, t, t0, t_M)

@njit
def get_CDS_Model(t_obs, t0, t_mat_grid, state_vec, lhc):
    n_mat, n_obs = t_mat_grid.shape
    CDS = np.ones((n_mat, n_obs))
    for mat_idx in range(n_mat):
        for i in range(n_obs):
            prot = psi_prot(lhc, t_obs[i], t0[i], t_mat_grid[mat_idx, i])
            prem = psi_prem(lhc, t_obs[i], t0[i], t_mat_grid[mat_idx, i])
            st = state_vec[:, i]
            CDS[mat_idx, i] = np.dot(prot, st) / np.dot(prem, st)
    return CDS

### NOTE: THIS METHODOLODY WILL NOT WORK, STILL NOT OPTIMIZING AT EACH STEP (SO USE PREVIOUS VAL)


class LHC_single():
    def __init__(self, r, delta, cds_tenor):
        # Set global params 
        self.r = r                  # Set short rate
        self.delta = delta          # Set recovery rate
        self.tenor = cds_tenor      # Set Swap tenor/payments structure.

    
    def initialise_LHC(self, Y_dim, X_dim, rng=None):
        if rng is None:
            rng = np.random.default_rng()  # independent each time

        self.Y_dim, self.m = Y_dim, X_dim
        self.a = np.ones((self.Y_dim,1))                                      # Y dim is 1 for LHC
        # Set inital values. Need to comply with (38)
        self.kappa = rng.uniform(0.1, 0.9, size=(X_dim,))       # Kappa given 
        self.gamma1 = np.array([np.min(self.kappa) - 1*10**(-6)])       # gamma1 strictly pos.
        self.theta = np.zeros(X_dim)

        for i in range(0,X_dim):
            self.theta[i] = rng.uniform(0, 1-self.gamma1/self.kappa[i], size=(1,))       # Theta coeffs
        # Build b, beta, A, gamma
        self.rebuild_dynamics()                                     # Build b,beta,gamma again.


    def rebuild_dynamics(self):
        # Formulas, cf p. 16.
        self.b = np.zeros(len(self.theta)).reshape((self.m, self.Y_dim))
        self.b[-1,:] = self.theta[-1] * self.kappa[-1]
        self.beta = np.zeros((self.m, self.m))
        for i in range(0,self.m):
            self.beta[i, i] = - self.kappa[i]
            if i + 1 < self.m:  
                self.beta[i, (i+1)] = self.kappa[i] * self.theta[i]
        # Build gamma. unit vec with gamma 1 in first entry.
        self.gamma = - np.array([[self.gamma1[0]] + [0] * (self.m - 1)]).reshape([1,self.m])

        # In LHC model, gamma is a row vector, b is a column vector for this to make senese.
        self.c = np.zeros(shape=(self.Y_dim, self.Y_dim))

        self.A = np.block([[self.c, self.gamma], 
                           [self.b, self.beta]])

        self.id_mat = np.identity(n=self.A.shape[0])
        self.A_star = self.A - self.r * self.id_mat
        self.A_star_inv = np.linalg.inv(self.A_star)


    def flatten_params(self):
        '''
        Extracting the parameters for optimization.
        '''
        self.shapes = {
            'kappa': self.kappa.shape,
            'theta': self.theta.shape,
            'gamma1': self.gamma1.shape,
        }
        return np.concatenate([
            self.kappa.flatten(),
            self.theta.flatten(),
            self.gamma1.flatten(),
        ])

    def unflatten_params(self, flat_vec):
        sizes = {k: np.prod(shape) for k, shape in self.shapes.items()}
        idx = 0
        for key in ['kappa', 'theta', 'gamma1']:
            size = sizes[key]
            shape = self.shapes[key]
            setattr(self, key, flat_vec[idx:idx + size].reshape(shape))
            idx += size
        self.rebuild_dynamics()
    

    def default_intensity(self,X,Y):
        # This is the form  of the LHC model
        return self.gamma1 * X[0,:]/Y

    def psi_Z(self, t, t_M):
        a_zeros = np.block([self.a, np.zeros(shape=self.m)])
        return (np.exp(-self.r * (t_M - t)) * a_zeros @ expm(self.A * (t_M - t))).ravel()

    def psi_D(self, t, t_M):
        mat_exp = expm(self.A_star * (t_M - t))
        c_gamma = np.block([self.c, self.gamma])
        return - (self.a.T @ c_gamma @ self.A_star_inv @ (mat_exp - self.id_mat)).ravel()

    def psi_D_star(self, t, t_M):
        mat_exp = expm(self.A_star * (t_M - t))
        c_gamma = np.block([self.c, self.gamma])
        return -(self.a.T @ c_gamma @ (
            (t_M - t)* self.A_star_inv @ mat_exp +
            self.A_star_inv @ (self.id_mat * t - self.A_star_inv) @ (mat_exp - self.id_mat)
        )).ravel()

    def psi_prot(self, t, t0, t_M):
        return (1 - self.delta) * (self.psi_D(t, t_M) - self.psi_D(t, t0))
    
    def psi_prem(self, t, t0, t_M):
        sum_Z = np.zeros(self.Y_dim + self.m)
        sum_D = np.zeros(self.Y_dim + self.m)
        t_grid_len = int(np.floor((t_M - t) / self.tenor).item()) + 1
        t_grid = np.zeros(t_grid_len)
        for i in range(t_grid_len):
            t_grid[i] = t + i * self.tenor
        for j in range(1, t_grid_len):
            dt = t_grid[j] - t_grid[j-1]
            sum_Z += dt * self.psi_Z(t, t_grid[j])
            if j < t_grid_len - 1:
                sum_D += dt * self.psi_D(t, t_grid[j])
        return (sum_Z + self.psi_D_star( t, t_M) - self.psi_D_star(t, t0)
                + t_grid[-2] * self.psi_D(t, t_M) - sum_D - t0 * self.psi_D( t, t0))

    def psi_cds(self, t, t0, t_M, k):
        return self.psi_prot(t, t0, t_M) - k * self.psi_prem(t, t0, t_M)


    def CDS_model(self,t_obs, T_M_grid, CDS_obs):
        # Get latent states.
        # X, Y, Z = self.get_states(t_obs, T_M_grid, CDS_obs)
        t0 = t_obs
        mat_actual = np.array([[0.2137 + i, 0.4658 + i,0.7178 + i,0.9671+i ] 
                                for i in range(0,int(np.max(t0)+1))]).flatten()
        # Ensure mat_actual is sorted
        mat_actual_sorted = np.sort(mat_actual)

        # For each element in t_mat_grid, find the smallest mat_actual that is >= element
        t0 = np.array([mat_actual_sorted[np.searchsorted(mat_actual_sorted, val, side='left')] 
                                        for val in t0.flatten()]).reshape(t0.shape)

        # Actual maturity dates. Say at March 20, Jun

        kappa, theta, gamma1 = self.kappa,self.theta,self.gamma1
        r = self.r
        Y_dim = self.Y_dim
        delta = self.delta
        tenor = self.tenor 
        lhc = rebuild_lhc_struct(kappa, theta, gamma1, r, Y_dim, delta, tenor)

        # New get states functionality:
        # Numba code to generate matrices to solve for.
        X,Y,Z = self.get_states(lhc, t_obs, T_M_grid, CDS_obs)

        #print('Done Getting Z,Y,X')
        state_vec = np.vstack([Y, X])

        # Here formula is the t_obs according to formula
        CDS = get_CDS_Model(t_obs, t_obs, T_M_grid, state_vec, lhc )
        #print('Done Getting CDS Rate')
        return CDS.T  # NOTE: CHANGED SIGN HERE. No idea why necessary.

    def get_states_out(self,t_obs, T_M_grid, CDS_obs):
        # Get latent states.
        t0 = t_obs
        mat_actual = np.array([[0.2137 + i, 0.4658 + i,0.7178 + i,0.9671+i ] 
                                for i in range(0,int(np.max(t0)+1))]).flatten()
        # Ensure mat_actual is sorted
        mat_actual_sorted = np.sort(mat_actual)

        # For each element in t_mat_grid, find the smallest mat_actual that is >= element
        t0 = np.array([mat_actual_sorted[np.searchsorted(mat_actual_sorted, val, side='left')] 
                                        for val in t0.flatten()]).reshape(t0.shape)

        kappa, theta, gamma1 = self.kappa,self.theta,self.gamma1
        r = self.r
        Y_dim = self.Y_dim
        delta = self.delta
        tenor = self.tenor 
        lhc = rebuild_lhc_struct(kappa, theta, gamma1, r, Y_dim, delta, tenor)

        # New get states functionality:
        # Numba code to generate matrices to solve for.
        X,Y,Z = self.get_states(lhc, t_obs, T_M_grid, CDS_obs)

        return X,Y,Z

    def get_states(self,lhc, t_obs, T_M_grid, CDS_obs):
        # RETHINK THIS A LOT. SEEMS LIKELY THAT THERE IS SOME SORT OF ERROR HERE. 
        n_obs = len(t_obs)
        n_mat = T_M_grid.shape[0]

        # Define initial values
        X0 = np.ones(shape=(lhc.m,)) *0.5
        X = np.ones((lhc.m, n_obs))
        Y = np.ones((n_obs)) # Implicitly sets Y0
        # Time 0 values
        X[:, 0] = X0
        # Previous Z, starting guess
        Z = np.ones((lhc.m,n_obs))
        Y_prev = Y[0]
        X_prev = X[:,0]
        Z_prev = X[:,0] / Y[0]


        ti = t_obs[1]
        ti_prev = t_obs[0]
        dt = ti-ti_prev
        # Find Z,X,Y
        for time_idx in range(n_obs):
            A_big = np.empty(shape = (n_mat,lhc.m))
            y_big = np.empty(shape = (n_mat))
            # Weight matrix
            #W = np.zeros(shape=(n_mat,n_mat))
            ti = t_obs[time_idx]
            # Build stacked vector
            one_Z = np.empty(shape = (1 + lhc.m))
            one_Z[0] = 1.0
            one_Z[1:] = Z_prev.flatten()
            
            for mat_idx in range(n_mat):
                psi_c = psi_cds(lhc,ti, ti, T_M_grid[mat_idx,time_idx], CDS_obs[time_idx,mat_idx])
                psi_p = psi_prem(lhc,ti, ti, T_M_grid[mat_idx,time_idx])
                d_k = np.dot(psi_p, one_Z)
                A_big[mat_idx,:] = - psi_c[1:] / d_k
                y_big[mat_idx] =  psi_c[0] / d_k    # Note, y needs to be negative to formulate as WLS problem
                #W[mat_idx,mat_idx] = 1 / d_k**2      # Needs to be squared to match reg .
            
            # Maybe not correct formulat (generalized inverse)
            if (lhc.m == 1) & (n_mat == 1):
                Z[:,time_idx] =  np.clip(y_big / A_big,0.0,1.0)
            else:
                # Only linear in states, NOT PARAMETers
                z = cp.Variable(self.m)
                objective = cp.Minimize(cp.sum_squares(A_big @ z - y_big))
                constraints = [0 <= z, z <= 1]
                prob = cp.Problem(objective, constraints)

                # Still dont get why not just least squares problem
                
                # The optimal objective value is returned by `prob.solve()`.
                try:
                    prob.solve(solver=cp.OSQP,verbose=True)
                except:
                    prob.solve(solver=cp.ECOS,verbose=True)      
                # The optimal value for x is stored in `x.value`.
                Z[:,time_idx] = np.clip(z.value,0.0,1.0)
            # Update Y and X
            Y[time_idx] = Y_prev + dt * (lhc.gamma.flatten() @ X_prev)
            X[:,time_idx] = Y[time_idx] * Z[:,time_idx]


            # Bump previous value
            Z_prev = Z[:,time_idx]
            X_prev = X[:,time_idx]
            Y_prev = Y[time_idx]


        return X,Y,Z

    
    def test_constriants(self):
        # Boolean mask of which constraints are satisfied
        satisfied = self.theta <= 1 - self.gamma1 / self.kappa

        # Indices that FAIL
        failed_idx = np.where(~satisfied)[0]   # ~ flips the booleans

        if failed_idx.size > 0:
            print(f"Constraint failed at indices: {failed_idx}")
            return False
        else:
             #print("All constraints satisfied.")
            return True
        
    def build_constraints(self, m):
        cons = []

        # g1 = x[i] * x[m+i] >= 0
        # for i in range(m):
        #     cons.append({'type': 'ineq',
        #                 'fun': lambda x, i=i: x[i] * x[m+i]})
        cons.append({'type': 'ineq',
                        'fun': lambda x: x[m] * x[2*m]})
        # g2: x[-1] - x[i] + x[i]*x[m+i] <= 0  ->  -g2 >= 0
        for i in range(m):
            cons.append({'type': 'ineq',
                        'fun': lambda x, i=i: -(x[-1] - x[i] + x[i]*x[m+i])})

        # Non-negativity
        for i in range(2*m+1):  # or len(x) if variable dimension
            cons.append({'type': 'ineq',
                        'fun': lambda x, i=i: x[i]})
        return cons



    ########### THIS METHODOLODY REBUILDS THE ONE IN ACKERER/FILIPOVIC. #################3
    def objective(self, x, t_obs, T_M_grid, CDS_obs, p = 0):
        # Format params for calculations
        # self.test_constriants()
        self.unflatten_params(x)

        #  Build Psi functions to avoid redoing it later.
        model_cds = self.CDS_model(t_obs, T_M_grid, CDS_obs)
        rmse = np.sqrt(np.mean((model_cds - CDS_obs)**2))

        penalty_func = 0
        # enforce constraint (1)
        for i in range(self.m):
            g1 = x[i]*x[self.m+i]
            penalty_func +=  max(0, -g1)**2
        # enforce constraint (2)
            g2 = x[-1] - x[i] + x[i]*x[self.m+i]
            penalty_func +=  max(0, g2)**2

        for i in range(len(x)):
            penalty_func +=  max(0, -x[i])**2

        # Non-negativity constraints (3)

        obj = rmse + p * penalty_func

        # print(f'Objective: {rmse}')
        # print(f'Parameters: {x}')
        
        return obj


    def optimize_params(self,t_obs, T_M_grid, CDS_obs,penalty=0):
        # Retrieve initial parameters. 
        flat_init = self.flatten_params().copy()
        CDS_obs = np.ascontiguousarray(CDS_obs)

        # result = minimize(
        #     fun = self.objective,
        #     x0 = flat_init,
        #     method = 'Nelder-Mead',
        #     args = (t_obs, T_M_grid, CDS_obs,penalty)
        # )
        constraints = self.build_constraints(self.m)

        result = minimize(
            fun=self.objective,
            x0=flat_init,
            args=(t_obs, T_M_grid, CDS_obs),
            method='SLSQP',
            constraints=constraints
            )

        if result.success:
            print(f"Optimization succeeded, params:{result.x}, objective: {result.fun}")
            self.unflatten_params(result.x)
            self.objective_result = result.fun
        else:
            print("Optimization failed:", result.message)
            self.unflatten_params(result.x)
            self.objective_result = result.fun

    def optimal_parameter_set(self, t_obs,T_M_grid, CDS_obs, penalty=0, base_seed = 1000,  n_restarts = 20):
        # Define grid of values. 
        current_objective = 1e10 #very high objective.
        out_params = self.flatten_params()
        for i in range(n_restarts):
            print(f"Optimization {i+1}")
            rng = np.random.default_rng(base_seed + i)  # deterministic but different
            self.initialise_LHC( self.Y_dim, self.m, rng=rng)
            self.optimize_params(t_obs, T_M_grid, CDS_obs,penalty)
            # Test new constraints
            constrains = self.test_constriants()

            if (self.objective_result < current_objective) & (constrains == True):
                print(f"New optimal parameters at iteration {i+1}.")
                current_objective = self.objective_result
                out_params = self.flatten_params()

        # Set new optimal parameters. 
        self.unflatten_params(out_params)


####################### KALMAN FILTERING OF THE ABOVE ########################



######### OPTION APPROXIMATION FORMULAS on credit default swaps.  ###############

    def get_bBounds(self, t0, t_M,k): 
        b_min = np.sum(np.minimum(self.psi_cds(t0,t0,t_M,k),np.zeros(self.m+1)))
        b_max = np.sum(np.maximum(self.psi_cds(t0,t0,t_M,k),np.zeros(self.m+1)))    

        return b_min,b_max
    

    def f_n(self,n,t,t0,b_min,b_max,Y):
        Lint, _ = quad(
            lambda x: (x * GenLegendrePoly(x,n,b_min,b_max)),
            0, b_max,
            limit=200,
            epsabs=1e-12,
            epsrel=1e-12
        )
        return np.exp(-self.r * (t0-t)) * Lint/ Y
    

    # Assume it is the inital price i need in Legendre poly.
    def PriceCDS(self,z, n,t,t0,t_M,k,Y):
        b_min,b_max = self.get_bBounds(t0, t_M,k)  

        pi = 0
        # Loop from 0 to n+1 (n)
        for j in range(n+1):
            f_n = self.f_n(j,t,t0,b_min,b_max,Y)
            GLPoly = GenLegendrePoly(z,j,b_min,b_max)
            pi += f_n * GLPoly
        
        return pi
    

## Pricing numba functions:
@njit
def LegendrePoly(x, n):
    # Compute standard Legendre. 
    Le0, Le1 = 1,x
    if n == 0:
        return Le0
    if n == 1:
        return Le1
    else:
        n_current = 1
        Le_np1 = 0 # Just start value.
        Le_n = Le1
        Le_nm1 = Le0
        while n_current < n:
            Le_np1 = (2*n_current + 1) * x * Le_n / (n_current + 1) - n_current * Le_nm1 / (n_current+1)
            # Bump values. 
            Le_nm1 = Le_n
            Le_n = Le_np1

            # Bump current. 
            n_current += 1

        return Le_np1

@njit
def GenLegendrePoly(x, n,b_min,b_max):
    mu = (b_max + b_min) / 2
    sigma = (b_max - b_min) / 2

    # TODO: Find out which n to use here in actuality.
    mathL = np.sqrt((1+2*n)/(2*sigma**2)) * LegendrePoly((x - mu) / sigma,n)

    return mathL