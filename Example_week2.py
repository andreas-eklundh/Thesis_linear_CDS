import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.integrate import quad
from Models.ATSM import ATSM

### Example in the end of week 2 slides. 
# set global params inspired by FID or the similar (ask chat)
sigma,a,b,c,rho = 0.1, 0.02,-2,0.01, -0.25 
# Somewhat oddly, the mean reversion param is b and long term mean a.
# This is due to us following the Filipovic notation. 
s0, r0 = 100, 0.01 # so lower interest rates than long term mean. 
# For now only do an Euler scheme (will also collide due to constant vols.)
T = 1 # maturity/expiry in a year. 
K = 100 # assume at-the-money initially. 
# Laplace complex integral
X0 = np.array([np.log(s0),r0])

### Try pricing using the ATSM class. 
K0 = np.array([-0.5*sigma**2,a])
K1 = np.array([[0,1],
               [0,b]])
H0 = np.array([[sigma**2, rho * sigma * c],
              [rho * sigma * c, c**2]])
H1 = np.array([[[0,0], [0,0]],
               [[0,0],[0,0]]])
rho0, rho1 = 0, np.array([0,1])
atsm = ATSM(K0,K1,H0,H1,rho0,rho1)

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
# Tested below w. high MC. Yields same result. 
# Conclusion: Usable. 


### MC simulation. 
# Simulate a path with number of datapoints N
def simulation_path(r0,s0,sigma,a,b,c,rho, delta,N):
    st, rt = np.ones(N)*np.log(s0),np.ones(N)*r0
    # simulate randomness. 
    w1 = norm.rvs(size=N)
    w2 = norm.rvs(size=N)
    z2 = rho * w1 + np.sqrt(1 - rho**2) * w2
    # loop for new instances.
    for i in range(1,N):
        # Get rt
        rt[i] = rt[i-1] + (a + b*rt[i-1])*delta + c*np.sqrt(delta)*z2[i]
        st[i] = st[i-1] + (rt[i]- 0.5 * sigma**2)*delta + sigma*np.sqrt(delta)*w1[i]
    return np.exp(st),rt

T,N = 1, 500
delta = T/N
s,r = simulation_path(r0,s0,sigma,a,b,c,rho, delta,N)
t_grid = np.array([i*delta for i in range(0,N)])
# plotting for fun. 
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axs[0].plot(t_grid, s, label='Stock Path')
axs[0].set_ylabel("Stock Price $S_t$")
axs[0].legend()
axs[0].grid()

axs[1].plot(t_grid, r, label='Interest Rate Path')
axs[1].set_ylabel("Interest Rate $r_t$")
axs[1].set_xlabel("Time")
axs[1].legend()
axs[1].grid()

plt.show()

# Obtain price of the option. 
def simulation_end(r0, s0, sigma, a, b, c, rho, delta, N):
    # Preallocate arrays
    w1 = norm.rvs(size=N)
    w2 = norm.rvs(size=N)
    z2 = rho * w1 + np.sqrt(1 - rho**2) * w2
    rt,r = r0, r0
    log_st,log_s = np.log(s0),np.log(s0)
    sum_rt = r0
    for i in range(1, N):
        r = rt + (a + b * rt) * delta + c * np.sqrt(delta) * z2[i]
        log_s = log_st + (r - 0.5 * sigma**2) * delta + sigma * np.sqrt(delta) * w1[i]
        sum_rt += r # Summing short rates for integration. 
        rt = r
        log_st = log_s

    return np.exp(log_st), sum_rt

# Compute price of option MC:
def price_call(r0,s0,sigma,a,b,c,rho, delta,K,N,M):
    prices = np.zeros(M)
    for i in range(0,M): 
        s,sum_rt = simulation_end(r0,s0,sigma,a,b,c,rho, delta,N)
        discount = sum_rt*delta
        prices[i] = np.exp(-discount)*np.max([0,s-K])
    return np.mean(prices)

price_MC =  price_call(r0,s0,sigma,a,b,c,rho, delta,K,N,M=100)
print(f"Price call MC {price_MC}")


### Calculate price of call using the MC approach. 
# For this to work, we need to solve the problem i.e.
# we need to do the heavy lifting of finding the discounted
# Laplace transform



# beta1 is w1 always as diffe eq. =0 and terminal cond. beta0=w
def beta1(T,w):
    return w[0] 

# beta_2:
#time t is time-to-mat. 
def beta2(T,w):
    return (beta1(T,w)*(np.exp(b*T)-1) + np.exp(b*T)*(b*w[1] - 1)+1)/b

# time t is time-to-mat.
def alpha(T,w):
    beta1_term = 0.5*sigma**2*beta1(T,w)*(beta1(T,w)-1)*T
    #beta2_int = np.sum(beta2(beta1, b,w,t_grid))*delta
    lower, upper = 0, T
    beta2_int_fixed = lambda t: beta2(t, w)
    beta2_int_fixed_sq = lambda t: beta2(t, w)**2

    beta2_int_re, _ = quad(lambda t: 
                        np.real(beta2_int_fixed(t)), lower,upper, limit=200, epsabs=1e-8, epsrel=1e-8)
    beta2_int_im, _ = quad(lambda t: 
                        np.imag(beta2_int_fixed(t)), lower,upper, limit=200, epsabs=1e-8, epsrel=1e-8)
    beta2_int_sq, _ = quad(lambda t: 
                        beta2_int_fixed_sq(t), lower,upper, limit=200, epsabs=1e-8, epsrel=1e-8)
    beta2_int_sq_re, _ = quad(lambda t: 
                        np.real(beta2_int_fixed_sq(t)), lower,upper, limit=200, epsabs=1e-8, epsrel=1e-8) 
    beta2_int_sq_im, _ = quad(lambda t: 
                        np.imag(beta2_int_fixed_sq(t)), lower,upper, limit=200, epsabs=1e-8, epsrel=1e-8) 
    beta2_int = beta2_int_re + 1j * beta2_int_im + w[1]
    beta2_int_sq = beta2_int_sq_re + 1j * beta2_int_sq_im + w[1]

    return beta1_term + (a+rho*sigma*c*beta1(T,w))*beta2_int+0.5*c**2*beta2_int_sq


def Laplace(w,X0,T):
    # Compute discounted Laplace
    A = alpha(T,w)
    B = np.array([beta1(T,w), beta2(T,w)])
    return np.exp(A + B.T @ X0)


def G_func(a_g, b_g, y, X0, T):
    first_term = Laplace(a_g, X0, T)
    Laplace_fixed = lambda v: Laplace(a_g + 1j * v * b_g, X0, T)
    
    upper_bound_int = 100
    Laplace_int, _ = quad(
        lambda v: np.imag(Laplace_fixed(v) * np.exp(-1j * v * y)) / v,
        0, upper_bound_int,
        limit=200,
        epsabs=1e-12,
        epsrel=1e-12
    )

    return first_term/2 - Laplace_int/np.pi


# Compute price:

price_laplace = G_func(np.array([1,0]),np.array([-1,0]),
                        -np.log(K),X0,T)-K*G_func(
                        np.array([0,0]),np.array([-1,0]),
                        -np.log(K),X0,T)
print(f"Price call Laplace {price_laplace}")


print(f"Summary: \n"
       f"MC price: {price_MC.round(3)}, \n"
       f"Laplace: {price_laplace.round(3)}, \n" 
       f"Laplace ATSModule {price_atsm.round(3)}")
print(f"Summary: \n"
       f"MC price: {price_MC.round(3)}, \n"
       f"Laplace: {price_laplace}, \n" 
       f"Laplace ATSModule {price_atsm}")

## Check of methods: 
vs = np.linspace(0.1, 100, 1000)
vals = [np.imag(atsm.Laplace_Transform(X0, np.array([0,0]) + 1j * v * np.array([-1,0]), T)) for v in vs]
vals2 = [np.imag( Laplace(np.array([0,0]) + 1j * v * np.array([-1,0]), X0, T)) for v in vs]
plt.plot(vs, vals, label = 'ATS-Module', color = 'red')
plt.plot(vs, vals2, label = 'Analytical', color = 'blue')

plt.title("Imaginary part of Laplace Transform vs v")
plt.xlabel("v")
plt.ylabel("Im[Laplace]")
plt.grid(True)
plt.show()