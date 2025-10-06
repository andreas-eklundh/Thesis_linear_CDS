#### Purpose: Try to see if prices/ payoffs from Filipovic Ackerer can be mimicked.

from Models.LHCModels.LHC_single import LHC_single
import numpy as np
import matplotlib.pyplot as plt
import os

# Initialise with values from the article.
lhc = LHC_single(r=0.0, delta=0.4,cds_tenor=0.25)
Y_dim,m = 1,1
X0 = 0.2
chi0 = np.array([1] + [X0]*m)
rng = np.random.default_rng(1000)
lhc.initialise_LHC(Y_dim,m,X0=X0,rng=rng)
lhc.flatten_params()
ell1,ell2 = 0.05,1
gamma1 = 0.25
beta = - (ell1+ell2)
b = ell1 * ell2 / gamma1
# Translate to kappa, theta also. Then beta=-kappa so kappa=(ell1+ell2). Theta is same ass b. normalizes by kappa
kappa = -beta
theta = b/kappa
lhc.flatten_params()

lhc.unflatten_params(np.array([kappa,theta,gamma1]))
sigma = 0.75


# Then pricing. 
t_start = 1
T_option = t_start + 5
strike_spreads = np.array([250,300,350]) / 10000
n_poly = np.array([1,5,30])
save_path = f"./Exploratory/"   # <--- change to your path

# Get time zero stats. 
M = 1000 # (how continous we make the plot)
for k in strike_spreads:
    fig, ax = plt.subplots(figsize=(10,6))
    # Get lower bounds on z.
    b_min,b_max = lhc.get_bBounds(t_start,T_option,k)
    # Create some grid fr plotting,
    plot_grid = np.array([b_min + i*(b_max-b_min)/M for i in range(0,M+1)])
    for n in n_poly:
        Y_t = chi0[0] 
        price = np.array([lhc.PriceCDS(z,n,t=0,t0=t_start,t_M=T_option,k=k,Y=Y_t) for z in plot_grid])
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
