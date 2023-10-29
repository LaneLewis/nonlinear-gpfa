import pyro
import pyro.distributions as dist
from gaussian_process import kernal_SE,zero_mean
import torch
from pyro.infer import MCMC, NUTS

class GPML_MV_Generating_Model():
    def __init__(self,embedding_func,taus,kernal_signal_sds,kernal_noise_sds,
                observation_noise):
        #taus:tensor 1 by latents
        #kernal_signal_sds 1 by latents
        #kernal_noise_sds 1 by latents
        #observation_noise 1 by observed
        #embedding_func f[] latents -> observed

        self.embedding_func = embedding_func
        self.taus = taus
        self.kernal_signal_sds = kernal_signal_sds
        self.kernal_noise_sds = kernal_noise_sds
        self.observation_noise = observation_noise
        self.observation_cov = torch.diag(self.observation_noise)
        self.vectorized_embedding_func = torch.vmap(embedding_func)
        self.embedding_dim = observation_noise.shape[0]
        self.latent_dim = self.taus.shape[0]

    def _flat_z_dist(self,times):
        #constructs the distribution for flattened zs
        Ks = []
        ms = []
        for i in range(self.latent_dim):
            Ks.append(kernal_SE(self.taus[i],self.kernal_signal_sds[i]**2,self.kernal_noise_sds[i]**2)(times))
            ms.append(zero_mean()(times))
        flat_z_block_cov = torch.block_diag(*Ks)
        flat_z_mean = torch.stack(ms)
        gp_prior = dist.MultivariateNormal(flat_z_mean,flat_z_block_cov,validate_args=False)
        return gp_prior
    
    def _flat_X_given_z_dist(self,z):
        time_steps = z.shape[0]
        #distribution of a single point mean/cov
        embedded_mean = self.vectorized_embedding_func(z)
        #gives the distribution for the flattened data
        flattened_cov = torch.block_diag(*[self.observation_cov]*time_steps)
        flattened_mean = torch.flatten(embedded_mean)
        X_flat_dist = dist.MultivariateNormal(flattened_mean,flattened_cov,validate_args=False)
        return X_flat_dist
    
    def sample_z(self,times):
        return pyro.sample("z_prior",self._flat_z_dist(times)).reshape((times.shape[0],self.latent_dim))
    
    def sample_X(self,times):
        X_data,_ = self.sample_joint(times)
        return X_data
    
    def sample_joint(self,times):
        gp_prior = self._flat_z_dist(times)
        z = pyro.sample("z",gp_prior).reshape((times.shape[0],self.latent_dim))
        X_flat_dist = self._flat_X_given_z_dist(z.reshape((times.shape[0],self.latent_dim,1)))
        X_flat_sample = pyro.sample("X_flat",X_flat_dist)
        X_data = torch.reshape(X_flat_sample,(times.shape[0],self.embedding_dim))
        return X_data,z
    
    def mcmc_sample_posterior(self,X,times,samples=1000,burn=10):
        def _sample_with_fixed_time(X):
            X_flat = torch.flatten(X)
            return pyro.poutine.condition(self.sample_X, data = {"X_flat": X_flat})(times)
        nuts_kernal = NUTS(_sample_with_fixed_time, jit_compile=True)
        mcmc = MCMC(nuts_kernal,num_samples=samples,warmup_steps=burn,num_chains=1,disable_progbar=True,)
        mcmc.run(X)
        return mcmc.get_samples()['z'].reshape((times.shape[0],self.latent_dim,samples))
    
    def joint_log_likelihood(self,X,z,times):
        z_ll = self._flat_z_dist(times).log_prob(z)
        X_flat = torch.flatten(X)
        X_ll = self._flat_X_given_z_dist(z).log_prob(X_flat)
        return z_ll+X_ll

    def joint_log_likelihood_given_z(self,X,times):
        def joint_log_likelihood(z):
            return self.joint_log_likelihood(X,z,times)
        return joint_log_likelihood
    
    def expectation_of_z_given_X(self,X,times,func_of_z="ll",samples=100,burn=10):
        if func_of_z == "ll":
            func_for_expectation = self.joint_log_likelihood_given_z(X,times)
        else:
            func_for_expectation = func_of_z
        mcmc_samples = self.mcmc_sample_posterior(X,times,samples,burn)
        vectorized_func_for_expectation = torch.vmap(func_for_expectation)

        average_likelihood = torch.mean(vectorized_func_for_expectation(mcmc_samples.reshape((samples,times.shape[0],1))))
        return average_likelihood
    