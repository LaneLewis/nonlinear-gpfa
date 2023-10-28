import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
import gaussian_process as gp
import torch
from MCMC_GP import simple_training_data
from pyro.infer import MCMC, NUTS
print(torch.__version__)
def simple_embedding(theta):
    def embed(z):
        time_length = z.shape[0]
        out_arr = torch.zeros((time_length,2),dtype=float)
        out_arr[:,0] = theta*z
        out_arr[:,1] = theta*z**2
        return out_arr
    return embed

tau = torch.tensor(0.5,dtype=float)
kernal_signal_sd = torch.tensor(0.1,dtype=float)
kernal_noise_sd = torch.tensor(0.1,dtype=float)

theta = torch.tensor(0.1,dtype=float)
observation_noise = torch.tensor([0.1,0.1],dtype=float)
times = torch.linspace(0.0,1.0,10,dtype=float)
embedding_func = simple_embedding(theta)

def generative_model(times,tau,kernal_signal_sd,kernal_noise_sd,
          embedding_func,observation_noise):
    gp_prior = latent_dist(times,tau,kernal_signal_sd,kernal_noise_sd)
    flat_X_given_z = flattened_X_given_z_dist(embedding_func,observation_noise)
    z = pyro.sample("z",gp_prior)
    X_flat_dist = flat_X_given_z(z)
    X_flat_sample = pyro.sample("X_flat",X_flat_dist)
    X_data = torch.reshape(X_flat_sample,(times.shape[0],observation_noise.shape[0]))
    return X_data

def flattened_X_given_z_dist(embedding_func,observation_noise):
    observation_cov = torch.diag(observation_noise)
    def X_given_z_dist(z):
        time_steps = z.shape[0]
        #distribution of a single point mean/cov
        embedded_mean = embedding_func(z)
        #gives the distribution for the flattened data
        flattened_cov = torch.block_diag(*[observation_cov]*time_steps)
        flattened_mean = torch.flatten(embedded_mean)
        X_flat_dist = dist.MultivariateNormal(flattened_mean,flattened_cov)
        return X_flat_dist
    return X_given_z_dist

def latent_dist(times,tau,kernal_signal_sd,kernal_noise_sd):
    K = gp.kernal_SE(tau,kernal_signal_sd**2,kernal_noise_sd**2)(times)
    m = gp.zero_mean()(times)
    gp_prior = dist.MultivariateNormal(m,K)
    return gp_prior

def joint_log_likelihood(X,z,flat_X_given_z_dist,z_dist):
    z_ll = z_dist.log_prob(z)
    X_flat = torch.flatten(X)
    X_ll = flat_X_given_z_dist(z).log_prob(X_flat)
    return z_ll+X_ll

def joint_log_likelihood_given_z(X,flat_X_given_z_dist,z_dist):
    def joint_log_likelihood(z):
        z_ll = z_dist.log_prob(z)
        X_flat = torch.flatten(X)
        X_ll = flat_X_given_z_dist(z).log_prob(X_flat)
        return z_ll+X_ll
    return joint_log_likelihood

def conditional_dist_generator(times,tau,kernal_signal_sd,kernal_noise_sd,
            embedding_func,observation_noise):
    def conditioned_dist(X):
        X_flat = torch.flatten(X)
        return pyro.poutine.condition(generative_model, data = {"X_flat": X_flat})(times,tau,kernal_signal_sd,kernal_noise_sd,
            embedding_func,observation_noise)
    return conditioned_dist

def sample_posterior(X,times,tau,kernal_signal_sd,kernal_noise_sd,
            embedding_func,observation_noise):
    conditional_model = conditional_dist_generator(times,tau,kernal_signal_sd,kernal_noise_sd,embedding_func,observation_noise)
    nuts_kernel = NUTS(conditional_model, jit_compile=True)
    mcmc = MCMC(nuts_kernel,num_samples=10,warmup_steps=10,num_chains=1,)
    mcmc.run(X)
    print(mcmc.summary(prob=0.5))
    return mcmc.get_samples()['z']

X,X_times,z,params = simple_training_data()
#model
z_dist = latent_dist(times,tau,kernal_noise_sd,kernal_noise_sd)
flat_X_given_z_dist = flattened_X_given_z_dist(embedding_func,observation_noise)
#likelihood
data_likelihood = joint_log_likelihood(X,z,flat_X_given_z_dist,z_dist)
posterior_samples = sample_posterior(X,times,tau,kernal_noise_sd,kernal_noise_sd,embedding_func,observation_noise)
print(posterior_samples.T)
vmap_likelihood = torch.vmap(joint_log_likelihood_given_z(X,flat_X_given_z_dist,z_dist))
vmap_likelihood(posterior_samples)
#joint_log_likelihood(X,)
#conditional_model = conditional_dist_generator(times,tau,kernal_signal_sd,kernal_noise_sd,
#          embedding_func,observation_noise)
#X_sample = generative_model(times,tau,kernal_signal_sd,kernal_noise_sd,embedding_func,observation_noise)
#nuts_kernel = NUTS(conditional_model, jit_compile=True)
#mcmc = MCMC(nuts_kernel,num_samples=10000,warmup_steps=1000,num_chains=1,)
#mcmc.run(X)
#mcmc.summary(prob=0.5)

