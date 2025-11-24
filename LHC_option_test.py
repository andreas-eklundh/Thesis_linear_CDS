#### Purpose: Try to see if prices/ payoffs from Filipovic Ackerer can be mimicked.
from Models.LHCModels.LHC_single import LHC_single,rebuild_lhc_struct
import numpy as np
import matplotlib.pyplot as plt
import os

# Initialise with values from the article.
r,delta,cds_tenor = 0.0,0.4,0.25
lhc = LHC_single(r=r, delta=delta,cds_tenor=cds_tenor)
Y_dim,m = 1,1
X0 = 0.2
chi0 = np.array([1] + [X0]*m)
rng = np.random.default_rng(1000)
lhc.initialise_LHC(Y_dim,m,X0=X0,rng=rng)
lhc.flatten_params()
ell1,ell2 = 0.05,1
gamma1 = 0.25
sigma = 0.75

# Use parameter values for first draft.
# ell1,ell2 = 0.03,0.2
# gamma1 = 0.10
# sigma = 0.20

beta = - (ell1+ell2)
b =  ell1 * ell2 / gamma1
# Translate to kappa, theta also. Then beta=-kappa so kappa=(ell1+ell2). Theta is same ass b. normalizes by kappa
kappa = -beta 
theta = b / kappa

lhc.unflatten_params(np.array([kappa,theta,gamma1]))
lhc.build_P_params()


# Then pricing. 
T_option = 1 # Maturity of option. Also the start of CDS
t_M = T_option + 5 # t_M = maturity of cds. 
# strike_spreads = np.array([250,300,350]) / 10000
strike_spreads = np.array([250]) / 10000
# strike_spreads = np.array([25,50,100]) / 10000

# Find model implied spread.
### Start by pricing option as in Filipovic Ackerer.
lhc.sigma = np.array([sigma])

# Test of other terms
psi_prot = lhc.psi_prot(0,0,t_M)
psi_prem = lhc.psi_prem(0,0,t_M)

psi_prot = lhc.psi_prot(0,T_option,t_M)
psi_prem = lhc.psi_prem(0,T_option,t_M)
print("chi0:", chi0)
print("psi_prot:", psi_prot)
print("psi_prem:", psi_prem)
print("psi_prot @ chi0:", float(psi_prot @ chi0))
print("psi_prem @ chi0:", float(psi_prem @ chi0))
print("model_spread (dec):", float((psi_prot @ chi0)/(psi_prem @ chi0)))
print("model_spread (bps):", float((psi_prot @ chi0)/(psi_prem @ chi0))*10000)
print("t, t0, tM:", 0, T_option, t_M)
print("lhc parameter vector (after unflatten):", lhc.get_params() if hasattr(lhc,'get_params') else "no get_params")
model_spread = (psi_prot @ chi0) / (psi_prem @ chi0)

lhc_tuple = rebuild_lhc_struct(lhc.kappa, lhc.theta, lhc.gamma1[0], lhc.r, lhc.Y_dim, lhc.delta, lhc.tenor)



print(f'Model spread: {model_spread*10000}')

price = lhc.get_cdso_price(t=0,t0=T_option,t_M=t_M,Y_t=1,X_t=X0,
                           strikes = strike_spreads,n_max=22) # 20 seems like enough
print(f'Price options strikes {strike_spreads*10000}, prices: {price*10000}')

N,M= 400000,100
cdso_MC_hist,cdso_MC = lhc.get_cdso_price_MC(0,T_option,t_M,strike_spreads,chi0,N,M,seed=1000)
print(f'MC: Price options strikes {strike_spreads*10000}, prices: {cdso_MC*10000}')

n_poly = np.array([1,5,30])


save_path = f"./Exploratory/"   # <--- change to your path

# Get time zero stats. 
M = 1000 # (how continous we make the plot)
for k in strike_spreads:
    fig, ax = plt.subplots(figsize=(10,6))
    # Get lower bounds on z - one for each strike.
    b_min,b_max = lhc.get_bBounds(T_option,t_M,k)
    # Create some grid fr plotting,
    plot_grid = np.array([b_min + i*(b_max-b_min)/M for i in range(0,M+1)])
    for n in n_poly:
        Y_t = chi0[0] 
        price = np.array([lhc.PriceCDS(z,n,t=0,t0=T_option,t_M=t_M,k=k,Y=Y_t) for z in plot_grid])
        ax.plot(10000*plot_grid,price*10000, label=f"Price CDSO, n={n}")
        print(f"Done with n={n},k={k}")
    ax.set_xlabel("z")
    ax.set_ylabel("Payoff")
    ax.set_title(f"Estimated CDSO Payoff, k={k}")
    ax.legend()
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f"Filipovic_CDSO_k_{k}.png"), dpi=150)
    plt.close(fig)

test=1
