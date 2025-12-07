from Models.BaselineCIR_alternative.CIR_Multifactor import CIRIntensity as CIR
import numpy as np

# seed = 37854
# cir = CIR(0.00248, 0.4, 0.25,X_dim=1)
# T = np.array([1,3,5,10])
# cir.set_params(params=None, seed=seed)
# print(cir.kappa,cir.theta,cir.sigma,cir.lambda1,cir.sigma_err)
# params_cir = np.concatenate([cir.kappa,cir.theta,cir.sigma,cir.lambda1,cir.sigma_err])
# x0 = np.zeros(cir.X_dim)
# alpha, beta = cir.cir_solution(params_cir,x0,T)
# cir.cascading = True
# alpha_n,beta_n =cir.cir_solution(params_cir,x0,T)

# print(f"Closed form: {alpha, beta} ")
# print(f"Numerical: {alpha_n,beta_n} ")

# alpha_x, beta_x = cir.cir_derivatives(params_cir,x0,T)
# cir.cascading = True
# alpha_x_n,beta_x_n =cir.cir_derivatives(params_cir,x0,T)


# print(f"Closed form: {alpha_x, beta_x} ")
# print(f"Numerical: {alpha_x_n,beta_x_n} ")

# Seems about right. Try Cascading model.
seed = 37854
cir = CIR(0.00248, 0.4, 0.25,X_dim=2)
T = np.array([1,3,5,10])
cir.set_params(params=None, seed=seed)
print(cir.kappa,cir.theta,cir.sigma,cir.lambda1,cir.sigma_err)
params_cir = np.concatenate([cir.kappa,cir.theta,cir.sigma,cir.lambda1,cir.sigma_err])
x0 = np.zeros(cir.X_dim)

alpha, beta = cir.cir_solution(params_cir,x0,T)
cir.cascading = True
alpha_n,beta_n =cir.cir_solution(params_cir,x0,T)

print(f"Closed form: {alpha, beta} ")
print(f"Numerical: {alpha_n,beta_n} ")

cir.cascading = False
alpha_x, beta_x = cir.cir_derivatives(params_cir,x0,T)
cir.cascading = True
alpha_x_n,beta_x_n = cir.cir_derivatives(params_cir,x0,T)


print(f"Closed form: {alpha_x, beta_x} ")
print(f"Numerical: {alpha_x_n,beta_x_n} ")

# Test mean

test = 1