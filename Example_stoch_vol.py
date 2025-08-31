import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal as mnorm
import Models
from Models.ATSM import ATSM

### Testing when adding jumps.

# Then pass its .pdf method

## Veryfied that it works on sumple stuff again. 
### Example on p. 17 on stochastic volatility models. 
a_s,a_v, b_s,b_v, c_v,c_s, h0,h1, rho = 0.02, 0,0.05,0.1,0.1,0.2,0.3,0.3, -0.3
S0, V0 = np.log(100), 0.01 # so lower interest rates than long term mean. 
# Note S is to be interpreted already as the log price. 
T = 1 # maturity/expiry in a year. 
K = 100 # assume at-the-money initially. 
# Laplace complex integral
X0 = np.array([S0,V0])

# Define matrices. 
K0 = np.array([a_s,a_v])
K1 = np.array([[b_s,0],
               [0,b_v]])
H0 = np.array([[c_s**2*h0,c_s*c_v*rho*h0 ],
              [c_s*c_v*rho*h0 , c_v**2*h0]])
H1 = np.array([[[0,c_s**2*h1], [0,c_s*c_v*rho*h1]],
               [[0,c_s*c_v*rho*h1],[0,c_v**2*h1]]])
# Model is constructed by assuming some relatinoship wrt. the short rate.
# Nothing given, just set rho0, to zero, while allowing for a 50% combination of each
rho0, rho1 = 0, np.array([0.5,0.5])
atsm = ATSM(K0,K1,H0,H1,rho0,rho1)

# Test pricing of a call option similar to the one in week2 example. 
price_atsm = atsm.G_transform(
                -np.log(K),a=np.array([1,0]),
                b=np.array([-1,0]),
                Xt=X0,T=T
                )- K*atsm.G_transform(
                -np.log(K),
                a=np.array([0,0]),
                b=np.array([-1,0]),
                Xt=X0,T=T
                )

print(f"Price call Laplace (ATSM-module) {price_atsm}")

## The simulate to see if approximately correct. 


# Obtain price of the option. 
def simulation_end(v0, s0, a_s,a_v,b_s, b_v,c_s, c_v,h0,h1,  rho,rho1, delta, N):
    # Preallocate arrays
    w1 = norm.rvs(size=N)
    w2 = norm.rvs(size=N)
    z2 = rho * w1 + np.sqrt(1 - rho**2) * w2
    vt,v = v0, v0
    st,s = s0,  s0
    sum_vt = np.matmul(rho1, np.array([s0,v0]))
    for i in range(1, N):
        v = vt + (a_v + b_v * vt) * delta + c_v * np.sqrt(h1*vt + h0) * np.sqrt(delta) * z2[i]
        s = st + (a_s + b_s * st)* delta + c_s * np.sqrt(h1*v + h0)* np.sqrt(delta) * w1[i]
        sum_vt += np.matmul(rho1, np.array([s,v])) # Summing short rates for integration. 
        vt = v
        st = s

    return st, sum_vt

# Compute price of option MC:
def price_call(v0, s0, a_s,a_v,b_s, b_v,c_s, c_v,h0,h1, rho,rho1, delta, N,M):
    prices = np.zeros(M)
    for i in range(0,M): 
        s,sum_vt = simulation_end(v0, s0, a_s,a_v,b_s, b_v,c_s, c_v,h0,h1,  rho,rho1, delta, N)
        discount = sum_vt*delta
        prices[i] = np.exp(-discount)*np.max([0,np.exp(s)-K])
        # print(i)
    return np.mean(prices)

N = 500 # mesh of 1000 points
delta = T/N
price_MC =  price_call(V0, S0, a_s,a_v,b_s, b_v,c_s, c_v,h0,h1,  rho,rho1, delta, N,M=10000)
print(f"Price call MC {price_MC}")


print(f"Summary: \n"
       f"MC price: {price_MC.round(3)}, \n"
       f"Laplace ATSModule {price_atsm.round(3)}")


