
import numpy as np
import sympy as sp
from Models.LHCModels.LHC_single import LHC_single 
from scipy.linalg import expm 
from Models.moments import MultivariatePolynomialGenerator as MPG
from Models.LHCModels.LHC_single import monomial_basis, rebuild_lhc_struct, moments_poly
### Test of monomial calcs. 
### Shape: (1,y,x_1,...,x_m,yx_1,yx2,...yxm,x_1^2, ..., x_m^n)
X_dim = 2
y,X = 0.9, np.array([0.5]*X_dim)
test,term_index = monomial_basis(y=y,X=X,poly_deg=1)

## Test of runctionality. 
lhc = LHC_single(0.025,0.4,0.25)
Y_dim,m = 1,X_dim
# Here, parameters are set already
rng = np.random.default_rng(1000)
lhc.initialise_LHC(Y_dim,m,X0=X,rng=rng)

chi = np.append(y,X)
moment_known = expm(lhc.A ) @ chi

T_to_mat = 1
# rebuild numba lhc
lhc_moment = rebuild_lhc_struct(lhc.kappa,lhc.theta,lhc.gamma1[0],
                                lhc.r,lhc.Y_dim,lhc.delta,lhc.tenor)
# moment Poly.
# Set sigma (not relevant for first order)
sigma = np.array([0.05]*m)
moment_poly = np.zeros(shape = lhc.Y_dim + lhc.m)
for i in range( lhc.Y_dim + lhc.m):
    # logic for better handling of indices...
    moment_poly[i] = moments_poly(y,X,lhc_moment,sigma,poly_deg=1,p_idxs=i+1,time_delta=T_to_mat)


### For the X and Y i can only approximate by simul - likely doable
### Fint covariance matrix. Need a somewhat special loop. Should subtract outer prod
# of the above.
moment_square = np.outer(moment_poly,moment_poly)

# Get second moments.
# Get index mapping.
second_moment = np.zeros(moment_square.shape)
_,idx = monomial_basis(y,X,poly_deg=2)
# Fill out upper triangular,add transpose or something.
# Write logic to simplify this...
## Covariance terms.
second_moment[0,1] = moments_poly(y,X,lhc_moment,sigma,poly_deg=2,p_idxs=idx['yX1'],time_delta=T_to_mat)
second_moment[0,2] = moments_poly(y,X,lhc_moment,sigma,poly_deg=2,p_idxs=idx['yX2'],time_delta=T_to_mat)
# second_moment[0,3] = moments_poly(y,X,lhc_moment,sigma,poly_deg=2,p_idxs=idx['yX3'],time_delta=T_to_mat)

second_moment[1,2] = moments_poly(y,X,lhc_moment,sigma,poly_deg=2,p_idxs=idx['X1X2'],time_delta=T_to_mat)
# second_moment[1,3] = moments_poly(y,X,lhc_moment,sigma,poly_deg=2,p_idxs=idx['X1X3'],time_delta=T_to_mat)
# second_moment[2,3] = moments_poly(y,X,lhc_moment,sigma,poly_deg=2,p_idxs=idx['X2X3'],time_delta=T_to_mat)
# Mirror above:
second_moment = second_moment + second_moment.T

# then second moments.
second_moment[0,0] = moments_poly(y,X,lhc_moment,sigma,poly_deg=2,p_idxs=idx['yy'],time_delta=T_to_mat)
second_moment[1,1] = moments_poly(y,X,lhc_moment,sigma,poly_deg=2,p_idxs=idx['X1X1'],time_delta=T_to_mat)
second_moment[2,2] = moments_poly(y,X,lhc_moment,sigma,poly_deg=2,p_idxs=idx['X2X2'],time_delta=T_to_mat)
# second_moment[3,3] = moments_poly(y,X,lhc_moment,sigma,poly_deg=2,p_idxs=idx['X3X3'],time_delta=T_to_mat)


# Still only diagonal second moment.
cov = second_moment - moment_square

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
sigma_ext = sp.Matrix([0, *lhc.sigma])   # prepend 0 symbolically
diag_entries = [sp.sqrt(chi_sym[i] * (chi_sym[0] - chi_sym[i])) * sigma_ext[i] for i in range(m+1)]
b = sp.diag(*diag_entries)

# As second term in drift, need monomials to fourth degree.
pmg = MPG(a, b, chi_sym, max_degree=2)

### Get moment matrices

G_n = pmg.generator_matrix()

### Expected value.
chi_dict = dict(zip(chi_syms, chi))
E_sympy = pmg.calculate_expected(chi_dict)
print(f'Known expectation: {moment_known}')
print(f'Sympy expectation {E_sympy}')

# Covariance
cov_symp = pmg.calculate_cov(chi_dict)

print(f'Manual Cov: {cov}')
print(f'Sympy Cov {cov_symp}')

stopper = 1