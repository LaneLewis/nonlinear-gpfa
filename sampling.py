import torch
from gaussian_process import kernal_SE
class Sampling():

    def __init__(self,embedding_func,taus,observation_noises,kernal_noise_sds,kernal_signal_sds):
        self.embedding_func = embedding_func
        self.latent_dims = taus.shape[0]
        self.observation_dims = observation_noises.shape[0]
        self.taus = taus
        self.kernal_noise_sds = kernal_noise_sds
        self.kernal_signal_sds = kernal_signal_sds
        self.observation_noises = observation_noises
        self.observation_cov = torch.diag(self.observation_noises)
        self.vectorized_embedding_func = torch.vmap(self.embedding_func)
        
    def sample_standard_normal(samples,dim):
        standard_normal_dist = torch.distributions.MultivariateNormal(torch.zeros((dim)),torch.eye(dim))
        return standard_normal_dist.sample((samples,))
    
    def sample_Z(self,samples,times):
        time_length = times.shape[0]
        Ks = []
        for i in range(self.latent_dims):
            Ks.append(kernal_SE(self.taus[i],self.kernal_signal_sds[i]**2,self.kernal_noise_sds[i]**2)(times))
        #construct the latent flat 
        flat_z_block_cov = torch.block_diag(*Ks).float()
        flat_z_mean = torch.zeros((time_length*self.latent_dims)).float()
        #sample from it
        prior_sampling_dist = torch.distributions.MultivariateNormal(flat_z_mean,flat_z_block_cov)
        #should have shape [samples,timelength*latents]
        prior_samples = prior_sampling_dist.rsample((samples,)).float()
        return prior_samples
    
    def sample_joint(self,samples,times):
        time_length = times.shape[0]
        Ks = []
        for i in range(self.latent_dims):
            Ks.append(kernal_SE(self.taus[i],self.kernal_signal_sds[i]**2,self.kernal_noise_sds[i]**2)(times))
        #construct the latent flat 
        flat_z_block_cov = torch.block_diag(*Ks).float()
        flat_z_mean = torch.zeros((time_length*self.latent_dims)).float()
        #sample from it
        prior_sampling_dist = torch.distributions.MultivariateNormal(flat_z_mean,flat_z_block_cov)
        #should have shape [samples,timelength*latents]
        prior_samples = prior_sampling_dist.rsample((samples,)).float()
        #should have the shape [samples, timelength, latents]
        reshaped_prior_samples = prior_samples.reshape(samples,time_length,self.latent_dims)
        #sampling from P(X|Z) - mu
        data_given_prior_mean = self.vectorized_embedding_func(reshaped_prior_samples)
        mean_centered_data_dist = torch.distributions.MultivariateNormal(torch.zeros(self.observation_dims),self.observation_cov)
        mean_centered_data_samples = mean_centered_data_dist.sample((samples,time_length))
        data = data_given_prior_mean + mean_centered_data_samples
        return data,reshaped_prior_samples
