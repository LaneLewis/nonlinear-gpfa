import pyro
import pyro.distributions as dist
from gaussian_process import kernal_SE,zero_mean
import torch
from pyro.infer import MCMC, NUTS
#todo: add multiple sample option for x and z
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
    
    def mcmc_sample_posterior(self,X,times,samples=1000,burn=10,divergence_ratio_warning=0.01,accept_prob = 0.80):
        def _sample_with_fixed_time(X):
            X_flat = torch.flatten(X)
            return pyro.poutine.condition(self.sample_X, data = {"X_flat": X_flat})(times)
        nuts_kernal = NUTS(_sample_with_fixed_time, target_accept_prob=accept_prob,jit_compile=True)
        mcmc = MCMC(nuts_kernal,num_samples=samples,warmup_steps=burn,num_chains=1,disable_progbar=True,)
        mcmc.run(X)
        divergences = len(mcmc.diagnostics()['divergences']['chain 0'])
        if divergences > 0:
            print("warning: Large Number of Sampling Divergences")
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
    
    def approx_elbo_loss(self,vi_means,vi_covs,X,times,sample_size=1000):
        '''Params:
            vi_means: python list (of size dim_latents) means with shape [time_dim]
            vi_covs: python list (of size dim_latents) cov with shape[time_dim,time_dim]
            X: tensor of shape [time_dim,neurons]
            times: tensor of shape [time_dim]
            sample_size: int specifying how many times to sample from the vi dist in order to approx the elbo'''
        #constructs the block normal dist for the prior
        time_length = times.shape[0]
        Ks = []
        ms = []
        for i in range(self.latent_dim):
            Ks.append(kernal_SE(self.taus[i],self.kernal_signal_sds[i]**2,self.kernal_noise_sds[i]**2)(times))
            ms.append(zero_mean()(times))
        flat_z_block_cov = torch.block_diag(*Ks).T
        flat_z_mean = torch.stack(ms).T
        #constructs the block normal dist for the vi
        flat_z_vi_mean = torch.stack(vi_means).T
        flat_z_vi_block_cov = torch.block_diag(*vi_covs).T
        #makes the kl_divergence term
        kl_term = normal_kl_divergence(flat_z_vi_mean,flat_z_vi_block_cov,flat_z_mean,flat_z_block_cov)
        #begins the section for sampling the p(x|z) term
        #prepares some terms that will be used many times 
        inv_observation_cov = torch.inverse(self.observation_cov)
        observation_det = torch.det(self.observation_cov)
        #samples from the Vi dist
        sampling_dist = torch.distributions.MultivariateNormal(flat_z_vi_mean.squeeze(),flat_z_vi_block_cov)
        #should have shape [samples,timelength*latents]
        samples = sampling_dist.sample((sample_size,))
        #should have the shape [samples, timelength, latents]
        reshaped_samples = samples.reshape(sample_size,time_length,self.latent_dim)
        #embeds the samples into the observation space mean. embedded_samples should have shape [samples, timelength, neurons]
        embedded_samples = self.vectorized_embedding_func(reshaped_samples)
        # broadcast X to embedded mean_samples this should have shape [samples, timelength, neurons]
        samples_mean_subtracted_X = X - embedded_samples
        #this should have shape [samples]
        log_likelihood_per_sample = batched_normal_dist_log_likelihood(samples_mean_subtracted_X,inv_observation_cov,observation_det)
        #final term
        approx_elbo = torch.mean(log_likelihood_per_sample) + kl_term
        return approx_elbo

def normal_kl_divergence(p_mean,p_cov,q_mean,q_cov):
    '''d(p||q)'''
    dim_p = p_mean.shape[0]
    dim_q = q_mean.shape[0]
    assert dim_p == dim_q
    q_cov_inv = torch.inverse(q_cov)
    det_term = torch.log(torch.det(q_cov)/torch.det(p_cov)) - dim_p
    quadratic_term = torch.linalg.multi_dot([(p_mean - q_mean).T,q_cov_inv,(p_mean - q_mean)])
    trace_term = torch.trace(torch.matmul(q_cov_inv,p_cov))
    return det_term + quadratic_term + trace_term

def batched_normal_dist_log_likelihood(batch_mean_subtracted_X,inv_observation_cov,observation_det):
    #batch_mean_subtracted_X has dim [samples, timelength, neurons]
    def normal_log_likelihood_over_time(mean_subtracted_X):
        time_dim = mean_subtracted_X.shape[0]
        neuron_dim = mean_subtracted_X.shape[1]
        const_term = neuron_dim*torch.log(torch.tensor(2*torch.pi))
        quadratic_term = torch.trace(torch.linalg.multi_dot([mean_subtracted_X,inv_observation_cov,mean_subtracted_X.T]))
        return -1/2*(time_dim*const_term + quadratic_term + time_dim*torch.log(observation_det))
    return torch.vmap(normal_log_likelihood_over_time,in_dims=0,out_dims=0)(batch_mean_subtracted_X)

