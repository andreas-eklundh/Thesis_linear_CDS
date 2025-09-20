import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from scipy.stats import norm
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import lsq_linear


class LHC_single():
    def __init__(self, r, delta, cds_tenor):
        # Set global params 
        self.r = r                  # Set short rate
        self.delta = delta          # Set recovery rate
        self.tenor = cds_tenor      # Set Swap tenor/payments structure.

    
    def initialise_LHC(self, Y_dim, X_dim, seed=100):
        '''
        Takes as input the dimension of teh two processes Y and X
        '''
        self.Y_dim, self.m = Y_dim, X_dim
        np.random.seed(seed)
        self.a = np.ones((self.Y_dim,1))                                      # Y dim is 1 for LHC
        # Set inital values. Need to comply with (38)
        self.kappa = np.random.uniform(0, 1, size=(X_dim,))       # Kappa given 
        self.gamma1 = np.array([np.min(self.kappa) - 1*10**(-3)])       # gamma1 strictly pos.
        self.theta = np.zeros(X_dim)
        # self.kappa = np.array([0.167,0.165])
        # self.gamma1 = np.array([0.056])
        # self.theta = np.array([0.666,0.662])

        for i in range(0,X_dim):
            self.theta[i] = np.random.uniform(0, 1-self.gamma1/self.kappa[i], size=(1,))       # Theta coeffs
        self.c = np.zeros(shape=(Y_dim, Y_dim))
        # Build b, beta, A, gamma
        self.rebuild_dynamics()                                     # Build b,beta,gamma again.


    def rebuild_dynamics(self):
        # Formulas, cf p. 16.
        self.b = np.block([np.zeros(len(self.theta[:-1])), self.theta[-1] * self.kappa[-1]]).reshape((self.m, self.Y_dim))
        self.beta = np.zeros((self.m, self.m))
        for i in range(0,self.m):
            self.beta[i, i] = - self.kappa[i]
            if i + 1 < self.m:  
                self.beta[i, (i+1)] = self.kappa[i] * self.theta[i]
        # Build gamma. unit vec with gamma 1 in first entry.
        self.gamma = - np.array([[self.gamma1[0]] + [0] * (self.m - 1)]).reshape([1,self.m])

        # In LHC model, gamma is a row vector, b is a column vector for this to make senese.

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

    def survival(self, Y_t):
        return self.a.T @ np.array([Y_t])

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

    def CDS_model(self, t_obs, T_M_grid, CDS_obs):
        # Get latent states.
        # NOTE ERROS EITHER IN GET STATE LOGIC.
        X, Y, Z = self.get_states(t_obs, T_M_grid, CDS_obs)
        print('Done Getting Z,Y,X')
        state_vec = np.vstack([Y, X])
        CDS = np.ones(( len(T_M_grid),len(t_obs)))
        t0 = t_obs

        # NOTE: OR ERROR IN PSI FUNCS.
        for mat_idx in range(len(T_M_grid)):
            for i in range(len(t_obs)):
                # state_vec = np.block([Y[i], X[:, i]])
                # t_mat = T_M_grid[mat_idx] + t_obs[i]
                nom = self.psi_prot(t_obs[i], t0[i], T_M_grid[mat_idx, i]) @ state_vec[:, i]
                denom = self.psi_prem(t_obs[i], t0[i], T_M_grid[mat_idx, i]) @ state_vec[:, i]
                CDS[mat_idx,i] =  nom / denom
        print('Done Getting CDS Rate')
        return -CDS.T  # NOTE: CHANGED SIGN HERE. No idea why necessary.

    def constraint_fun(self,x):
        kappa = x[:self.m]
        theta = x[self.m:2*self.m]
        gamma1 = x[-1]

        return 1 - gamma1 / kappa - theta   # vector of length m


    # def objective(self, x, t_obs, T_M_grid, CDS_obs,p=None):
    #     # Format params for calculations
    #     print('Trying new')
    #     self.unflatten_params(x)
    #     model_cds = self.CDS_model(t_obs, T_M_grid, CDS_obs)
    #     rmse = np.sqrt(np.mean((model_cds - CDS_obs)**2))
    #     print(f'Objective RMSE: {rmse}')
    #     # print(f'Parameters: {x}')
        
    #     return rmse
    
    def test_constriants(self):
        # Boolean mask of which constraints are satisfied
        satisfied = self.theta <= 1 - self.gamma1 / self.kappa

        # Indices that FAIL
        failed_idx = np.where(~satisfied)[0]   # ~ flips the booleans

        if failed_idx.size > 0:
            print(f"Constraint failed at indices: {failed_idx}")
        else:
            print("All constraints satisfied.")
        

    def objective(self, x, t_obs, T_M_grid, CDS_obs, p = 0):
        # Format params for calculations
        self.test_constriants()
        self.unflatten_params(x)
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

        print(f'Objective: {rmse}')
        print(f'Parameters: {x}')
        
        return obj


    def optimize_params(self,t_obs, T_M_grid, CDS_obs,penalty=0):
        # Retrieve initial parameters. 
        flat_init = self.flatten_params().copy()

        # cons = NonlinearConstraint(lambda x: self.constraint_fun(x), 0, np.inf)

        # # positivity constraints: bounds
        # bounds = [(1e-8, None)] * (2*self.m+1)  # all vars > 0

        # result = minimize( fun = self.objective,
        #                 x0 = flat_init, method="trust-constr",
        #                 constraints=[cons], bounds=bounds,
        #                 args = (t_obs, T_M_grid, CDS_obs,penalty))

        result = minimize(
            fun = self.objective,
            x0 = flat_init,
            method = 'Nelder-Mead',
            args = (t_obs, T_M_grid, CDS_obs,penalty)
        )
        if result.success:
            print("Optimization succeeded.")
            self.unflatten_params(result.x)
        else:
            print("Optimization failed:", result.message)

    # Obtain Z process (X normalized by Y):

    def get_states(self, t_obs, T_M_grid, CDS_obs):
        # RETHINK THIS A LOT. SEEMS LIKELY THAT THERE IS SOME SORT OF ERROR HERE. 
        n_obs = len(t_obs)
        n_mat = T_M_grid.shape[0]

        # Define initial values
        X0 = np.ones(shape=(self.m,)) *0.5
        X = np.ones((self.m, n_obs))
        Y = np.ones((n_obs)) # Implicitly sets Y0
        # Time 0 values
        X[:, 0] = X0
        # Previous Z, starting guess
        Z = np.ones((self.m,n_obs))
        Y_prev = Y[0]
        X_prev = X[:,0]
        Z_prev = X[:,0] / Y[0]


        ti = t_obs[1]
        ti_prev = t_obs[0]
        dt = ti-ti_prev
        # Find Z,X,Y
        for time_idx in range(n_obs):
            A_big = np.empty(shape = (n_mat,self.m))
            y_big = np.empty(shape = (n_mat))
            # Weight matrix
            W = np.zeros(shape=(n_mat,n_mat))
            ti = t_obs[time_idx]
            # Build stacked vector
            one_Z = np.empty(shape = (1 + self.m))
            one_Z[0] = 1.0
            one_Z[1:] = Z_prev.flatten()

            for mat_idx in range(n_mat):
                psi_c = self.psi_cds(ti, ti, T_M_grid[mat_idx,time_idx], CDS_obs[time_idx,mat_idx])
                psi_p = self.psi_prem(ti, ti, T_M_grid[mat_idx,time_idx])
                d_k = np.dot(psi_p, one_Z)
                A_big[mat_idx,:] = - psi_c[1:] 
                y_big[mat_idx] =  psi_c[0]      # Note, y needs to be negative to formulate as WLS problem
                W[mat_idx,mat_idx] = 1 / d_k**2      # Needs to be squared to match reg .
            
            # Maybe not correct formulat (generalized inverse)
            sol = lsq_linear(A_big, y_big, bounds=(0, 1), method="trf")  # BVLS/trust-region reflective
            Z[:,time_idx]  = sol.x

            # Update Y and X
            Y[time_idx] = Y_prev + dt * (self.gamma.flatten() @ X_prev)
            X[:,time_idx] = Y[time_idx] * Z[:,time_idx]


            # Bump previous value
            Z_prev = Z[:,time_idx]
            X_prev = X[:,time_idx]
            Y_prev = Y[time_idx]


        return X,Y,Z
