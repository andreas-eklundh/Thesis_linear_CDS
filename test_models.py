
import numpy as np
import sympy as sp
from Models.LHCModels.LHC_single import LHC_single 
from scipy.linalg import expm 
from Models.moments import MultivariatePolynomialGenerator as MPG
from Models.LHCModels.LHC_single import rebuild_lhc_struct


#### Numerical test of CIR example.
kappa,theta,sigma = 0.2,0.05,0.1 
X0=np.array([0.07])
T,t=3,0
delta = T-t

# Values.
E_theo = (theta * (1-np.exp(-kappa * delta)) +
                        X0 * (np.exp(-kappa * delta)))
           
cov_theo = (sigma**2 * theta * (1-np.exp(-kappa * delta))**2 / (2 * kappa) +
                        X0 * sigma**2 * (np.exp(-kappa * delta) - np.exp(-2*kappa * delta))/ kappa)
                


chi_syms = sp.symbols(['X'])   
chi_sym = sp.Matrix(chi_syms)               
drift = ( sp.Matrix([kappa*theta])  - kappa*chi_sym)
diffusion =  sigma * sp.sqrt(chi_sym)   # prepend 0 symbolically

# As second term in drift, need monomials to fourth degree.
pmg = MPG(drift, diffusion, chi_sym, max_degree=2)

### Get moment matrices

G_n = pmg.generator_matrix()
chi_dict = dict(zip(chi_syms, X0))
E_sympy = pmg.calculate_expected(chi_dict,delta)
print(f'Actual expectation {E_theo}')
print(f'Sympy expectation {E_sympy}')

# Covariance
cov_symp = pmg.calculate_cov(chi_dict,delta)
print(f'Actual Cov {cov_theo}')
print(f'Sympy Cov {cov_symp}')




##### LHC model moments

### Test of monomial calcs. 
### Shape: (1,y,x_1,...,x_m,yx_1,yx2,...yxm,x_1^2, ..., x_m^n)
X_dim = 1
y,X = 1, np.array([0.07]*X_dim)
## Test of runctionality. 
lhc = LHC_single(0.025,0.4,0.25)
Y_dim,m = 1,X_dim
# Here, parameters are set already
rng = np.random.default_rng(1000)
lhc.initialise_LHC(Y_dim,m,X0=X,rng=rng)
# Set parameters

# Test of params. 
print(f'thetakappa>=sigma**2/2: {kappa*theta}>={sigma**2/2}')
gamma1 = 0.1
print(f'Other drift cond: {gamma1 - kappa + kappa*theta} <= {-sigma**2/2}')

params = np.array([kappa,theta,gamma1])
lhc.flatten_params()
lhc.unflatten_params(params)

t_diff  = 3

chi = np.append(y,X)
moment_known = expm(lhc.A * t_diff ) @ chi


# rebuild numba lhc
lhc_moment = rebuild_lhc_struct(lhc.kappa,lhc.theta,lhc.gamma1[0],
                                lhc.r,lhc.Y_dim,lhc.delta,lhc.tenor)

test,term_index = lhc.monomial_basis(y=y,X=X,poly_deg=1)

# moment Poly.


### Simulate to compare...
# Set also P params. 
lhc_P = lhc.build_P_params(rng=rng)
lhc.sigma = sigma

# Simulate. We are using an Euler discretization. 
# Start at 0.5 also for aloowing for more jump op and down. Again, likly too large initial cov
# alphas = compute_enumerations(X_dim,n=1)
# E_Z = moments_Z(lhc_moment,alphas,t0=0,t_M=1,k=0.05,y=y,X=X,n=1)
T,M = 1, 500
# Use same seed to reproduce same randomness.
# mat_grid = np.array([1,3,5,7,10]) # Typical maturity grid
# mat_grid = np.array([1]) # not important for simul 

# n_mat = mat_grid.shape[0]
# N = 100
# chi_end = np.zeros((lhc.m+1,N))
# chi_end_mil = np.zeros((lhc.m+1,N))

# for i in range(0,N):
#     T_path, chi_Q = lhc.simul_latent_states(chi0=chi,T=T,M=M,n_mat=1,scheme='Euler')
#     T_path, chi_Q_mil = lhc.simul_latent_states(chi0=chi,T=T,M=M,n_mat=1,scheme='Milstein')
#     chi_end[:,i] = np.array(chi_Q[-1,:])
#     chi_end_mil[:,i] = np.array(chi_Q_mil[-1,:])

# # Estimated mean:
# print(f'Simulated mean Euler: {np.mean(chi_end,axis=1)}')
# print(f'Simulated mean Milstein: {np.mean(chi_end_mil,axis=1)}')
# print(f'Known expectation: {moment_known}')
# print(f'Poly expectaiton {moment_poly}')

# cov_matrix = np.cov(chi_end)  # rowvar=False because columns are variables
# cov_matrix_mil = np.cov(chi_end_mil)  # rowvar=False because columns are variables    

# print(f'Euler cov: {cov_matrix}')
# print(f'Milstein Cov: {cov_matrix_mil}')
# print(f'Actual: {cov}')


### Try out new logic.
# Play around with z=1 instance. 

### Test of logic agains previous.

# Create symbols automatically
chi_syms = sp.symbols(f'Y X1:{m+1}')   
chi_sym = sp.Matrix(chi_syms)               

a = lhc.A @ chi_sym
sigma_ext = sp.Matrix([0, lhc.sigma])   # prepend 0 symbolically
diag_entries = [sp.sqrt(chi_sym[i] * (chi_sym[0] - chi_sym[i])) * sigma_ext[i] for i in range(m+1)]
b = sp.diag(*diag_entries)

# As second term in drift, need monomials to fourth degree.
pmg = MPG(a, b, chi_sym, max_degree=2)

### Get moment matrices

G_n = pmg.generator_matrix()

### Expected value.
chi_dict = dict(zip(chi_syms, chi))
E_sympy = pmg.calculate_expected(chi_dict,t_diff)

# Covariance
cov_symp = pmg.calculate_cov(chi_dict,t_diff)


### Finally, we do the implementation manually as well.

G_manual = np.array([[0,0,0,0,0,0],
                        [0,0,lhc.b.flatten()[0],0,0,0],
                        [0,lhc.gamma.flatten()[0],lhc.beta.flatten()[0],0,0,0],
                        [0,0,0,0,lhc.b.flatten()[0],0],
                        [0,0,0,2*lhc.gamma.flatten()[0],lhc.beta.flatten()[0],2*lhc.b.flatten()[0] + lhc.sigma**2],
                        [0,0,0,0,lhc.gamma.flatten()[0], 2*lhc.beta.flatten()[0] - lhc.sigma**2]])


### Then go through a lot of indicator stuff.
basis_manual = np.array([1,y,X[0],y**2,X[0]*y,X[0]**2])
moment_manual = np.zeros(6)
for i in range(0,6):
    e_i = np.zeros(6)
    e_i[i] = 1
    moment_manual[i] = basis_manual.T @ expm(G_n*t_diff) @ e_i


print(f'Known expectation: {moment_known}')
print(f'Manual expectation {moment_manual[1:3]}')
print(f'Sympy expectation {E_sympy}')

# Covariance
cov_symp = pmg.calculate_cov(chi_dict,t_diff)

### Govariance manual
cov_man = np.array([[moment_manual[3]-moment_manual[1]**2,moment_manual[4]-moment_manual[1]*moment_manual[2] ],
                    [moment_manual[4]-moment_manual[1]*moment_manual[2], moment_manual[5]-moment_manual[2]**2] ])

print(f'Sympy Cov {cov_symp}')
print(f'Manual Cov {cov_man}')


stopper = 1