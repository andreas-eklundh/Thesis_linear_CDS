import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.integrate import solve_ivp

class Kalman_filter():
    def __init__(self):
        return None

    # Estimate latent states using optimal Kalman.
    def run_Kalman(self,m1,P1,A,b,d,Q,H,R,data):
        mn = np.zeros((m1.shape[0],data.shape[0]))
        Pn = np.zeros((P1.shape[0],P1.shape[0]))
        # Store predictions. 
        pred_mn = np.zeros((m1.shape[0],data.shape[0]))
        pred_Pn = np.zeros((P1.shape[0],P1.shape[0],data.shape[0]))
        # We want to store all predictions. 
        pred_mn[:,0]  = m1
        pred_Pn[:,:,0]  = P1

        log_likelihood = 0

        # Loop over index timepoints
        for n in range(0,data.shape[0]):
            # UPDATE STEP
            vn = data[n,:] - (H @ pred_mn[:,n] + d)
            Sn = H @ pred_Pn[:,:,n] @ H.T + R
            # Compute Sn_inv - using proper inv or fallback on pseudo
            # bad practice, bud fine for this purpose.
            try: 
                Sn_inv = np.linalg.inv(Sn) 
            except:
                Sn_inv = np.linalg.pinv(Sn)
            Kn = pred_Pn[:,:,n] @ H.T @ Sn_inv 
            mn[:,n] = pred_mn[:,n] + Kn @ vn
            Pn = pred_Pn[:,:,n] - Kn @ Sn @ Kn.T    

            # Prediction step
            if (n < data.shape[0] - 1): # Not sensible to predict further.
                pred_mn[:,n+1] = A @ mn[:,n] + b # Note initial guess overwritten
                pred_Pn[:,:,n+1] = A @ Pn @ A.T + Q
                # Return log likelihood as well.         

            # det_Sn = np.abs(np.linalg.det(Sn))
            det_Sn = np.abs(np.linalg.det(Sn))

            log_likelihood += - Sn.shape[0]/2 * np.log(2*np.pi) - 1/2 * np.log(det_Sn) - 1/2 * vn.T @ Sn_inv @ vn

        return mn, pred_mn, pred_Pn, log_likelihood
    
    # One for doing the optimization.
    def estimate_params(self):
        return None 