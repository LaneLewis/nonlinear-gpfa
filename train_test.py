import torch
import numpy as np
from gaussian_process import gaussian_process_prior,kernal_SE,zero_mean

def make_embedded_alternating_square_data(sigmas,tau=0.5,signal_var=0.01,noise_var=0.01,
                                          time_begin=0.0,time_end=10.0,timesteps=1000,embedding_dim=10):
    times = torch.tensor(np.linspace(time_begin,time_end,timesteps))
    K = kernal_SE(tau,signal_var,noise_var)
    m = zero_mean()
    gp_prior = gaussian_process_prior(times,m,K)
    z_true = gp_prior.rsample()
    square_embedding = alternating_squared_embedding(z_true,embedding_dim)
    return add_ind_row_noise(square_embedding,sigmas)

def alternating_squared_embedding(z_vec,embedding_dim):
    #takes in a single z_vec
    z_squared_vec = torch.square(z_vec)
    squared_zs_number = embedding_dim//2
    squared_zs = torch.outer(z_squared_vec,torch.ones(squared_zs_number))
    linear_zs = torch.outer(z_vec,torch.ones(embedding_dim - squared_zs_number))
    return torch.concat((squared_zs,linear_zs),dim=1)

def add_ind_row_noise(dataset,sigmas):
    dataset_rows = dataset.shape[0]
    dataset_cols = dataset.shape[1]
    noise_dist = torch.distributions.MultivariateNormal(torch.zeros(dataset_cols),torch.diag(sigmas))
    noise_samples = noise_dist.sample((dataset_rows,))
    return dataset+noise_samples

make_embedded_alternating_square_data(0.5*torch.ones(10))
