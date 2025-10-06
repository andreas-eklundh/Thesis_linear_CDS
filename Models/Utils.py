import numpy as np


### We look into fit on CDS spreads as these are the ones we effectively needs to produce. 

class global_fit_measures:
    def __init__(self,obs,models):
        '''
        obs: The observes timeseries of e.g. CDS spreads. 
        models: A list of model/fitted e.g. CDS spreads.
        '''

        self.obs = obs
        self.models = models

        # We also set some global parameters needed in each calculation
        self.n_mat = self.obs.shape[1] # Corresponding to n_options in article? 
        self.n_models = len(self.models)
        self.n_obs = self.obs.shape[0]
    # Each function should just return history and table size.

    # Root mean squared error. 
    def rmse(self):
        rmse_series = np.zeros((self.n_obs,self.n_models))
        rmse = np.zeros((self.n_models))
        for i in range(self.n_models):
            diff = (self.obs - self.models[i])**2
            rmse[i] = np.sqrt(np.sum(diff)/(self.n_mat*self.n_obs))
            rmse_series[:,i] = np.sqrt(np.sum(diff,axis=1)/(self.n_mat))

            
        return rmse_series, rmse

    # Average Absolite error as percentage of mean price:
    def ape(self):
        mean = np.mean(self.obs)
        mean_series = np.mean(self.obs,axis=1)
        ape = np.zeros(self.n_models)
        ape_series = np.zeros((self.n_obs,self.n_models))
        for i in range(self.n_models):
            diff = np.abs(self.obs - self.models[i])
            ape[i] = np.sum(diff)/(self.n_mat*self.n_obs) * 1/mean
            ape_series[:,i] = np.sum(diff,axis=1)/(self.n_mat) * 1/mean_series
        return ape_series, ape

    #Average absolute error:
    def aae(self):
        aae = np.zeros(self.n_models)
        aae_series =  np.zeros((self.n_obs,self.n_models))
        for i in range(self.n_models):
            diff = np.abs(self.obs - self.models[i])
            aae[i] = np.sum(diff)/(self.n_mat*self.n_obs)
            aae_series[:,i] = np.sum(diff,axis=1)/(self.n_mat)

        return aae_series,aae

    #Average relative percentage error:
    def arpe(self):
        arpe = np.zeros(self.n_models)
        arpe_series = np.zeros((self.n_obs,self.n_models))
        for i in range(self.n_models):
            diff = np.abs(self.obs - self.models[i]) / self.obs
            arpe[i] = np.sum(diff)/(self.n_mat*self.n_obs)
            arpe_series[:,i] = np.sum(diff,axis=1)/(self.n_mat)

        return arpe_series,arpe