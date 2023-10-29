import pyro
import pyro.distributions as dist
from gaussian_process import kernal_SE,zero_mean
import torch
from pyro.infer import MCMC, NUTS

class GPML_Generating_Model():
    def __init__(self,embedding_func,tau,kernal_signal_sd,kernal_noise_sd,
                observation_noise):
        self.embedding_func = embedding_func
        self.tau = tau
        self.kernal_signal_sd = kernal_signal_sd
        self.kernal_noise_sd = kernal_noise_sd
        self.observation_noise = observation_noise
        self.observation_cov = torch.diag(self.observation_noise)
        self.vectorized_embedding_func = torch.vmap(embedding_func)
        self.embedding_dim = observation_noise.shape[0]

    def z_dist(self,times):
        K = kernal_SE(self.tau,self.kernal_signal_sd**2,self.kernal_noise_sd**2)(times)
        m = zero_mean()(times)
        gp_prior = dist.MultivariateNormal(m,K,validate_args=False)
        return gp_prior
    
    def flat_X_given_z_dist(self,z):
        time_steps = z.shape[0]
        #distribution of a single point mean/cov
        embedded_mean = self.vectorized_embedding_func(z)
        #gives the distribution for the flattened data
        flattened_cov = torch.block_diag(*[self.observation_cov]*time_steps)
        flattened_mean = torch.flatten(embedded_mean)
        X_flat_dist = dist.MultivariateNormal(flattened_mean,flattened_cov,validate_args=False)
        return X_flat_dist
    
    def sample_z(self,times):
        return pyro.sample("z_prior",self.z_dist(times)).reshape((times.shape[0],1))
    
    def sample_X(self,times):
        X_data,_ = self.sample_joint(times)
        return X_data
    
    def sample_joint(self,times):
        gp_prior = self.z_dist(times)
        z = pyro.sample("z",gp_prior).reshape((times.shape[0],1))
        X_flat_dist = self.flat_X_given_z_dist(z)
        X_flat_sample = pyro.sample("X_flat",X_flat_dist)
        X_data = torch.reshape(X_flat_sample,(times.shape[0],self.embedding_dim))
        return X_data,z
    
    def mcmc_sample_posterior(self,X,times,samples=1000,burn=10):
        def _sample_with_fixed_time(X):
            X_flat = torch.flatten(X)
            return pyro.poutine.condition(self.sample_X, data = {"X_flat": X_flat})(times)
        nuts_kernal = NUTS(_sample_with_fixed_time)
        mcmc = MCMC(nuts_kernal,num_samples=samples,warmup_steps=burn,num_chains=1,disable_progbar=True,)
        mcmc.run(X)
        return mcmc.get_samples()['z']
    
    def joint_log_likelihood(self,X,z):
        z_ll = self.z_dist.log_prob(z)
        X_flat = torch.flatten(X)
        X_ll = self.flat_X_given_z_dist(z).log_prob(X_flat)
        return z_ll+X_ll

    def joint_log_likelihood_given_z(self,X,times):
        def joint_log_likelihood(z):
            z_ll = self.z_dist(times).log_prob(z)
            X_flat = torch.flatten(X)
            X_ll = self.flat_X_given_z_dist(z).log_prob(X_flat)
            return z_ll+X_ll
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
    