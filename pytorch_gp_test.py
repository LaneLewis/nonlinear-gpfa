import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

#nonlinear manifold embedding example dataset.
#parameter u follows a sin function over time and is embedded into R2
#with the nonlinear function [u^2,u]
def imbedded_sin_bowl(time_start,time_end,timesteps,sin_std,add_noise=True):
    time_steps = torch.linspace(time_start,time_end,timesteps)
    sin_func = torch.sin(time_steps) 
    if add_noise:
        noise = torch.tensor(np.random.normal(0,sin_std,sin_func.size()),dtype=torch.float)
        sin_func = sin_func + noise
    bowl_noisy_sin = sin_func**2
    return sin_func,bowl_noisy_sin

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# (should) take in an input of size T (time) by N (neurons)
# and compress it into an initial latent state l_0 of size equal to the 
# number of latents. This should then be unfolded over time into a latent
# dataset of size T by num_latents*(1 + covariance_rank). The first coordinate in each
# latent set corresponds to the mean and the remaining are parameters for the covariance.
# The latent set is then sampled from on each latent using N(mu,sum (sigma_i sigma_i^T)).
# these samples are collected and then used to calculate the loss.
class BiLSTM(nn.Module):
    def __init__(self,input_size,num_latents=1,covariance_rank=2,epsilon_samples=100):
        super().__init__()
        self.encoder_step_1 = nn.LSTM(input_size,num_latents + num_latents*covariance_rank)
        self.encoder_step_2 = nn.LSTM(num_latents + num_latents*covariance_rank,num_latents+num_latents*covariance_rank)
        self.decoder = nn.Sequential(nn.Linear(num_latents),
                                    nn.ReLU(),
                                    nn.Linear(num_latents, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, input_size))
        self.num_latents = num_latents
        self.covariance_rank = covariance_rank
        self.epsilon_samples = epsilon_samples
    def forward(self,x):
        time_steps = len(x)
        hidden_states,_ = self.encoder_step_1(x)
        initial_latent_states = torch.unsqueeze(hidden_states[-1],0).T
        zero_input_states = torch.zeros((initial_latent_states.shape[0],len(x)-1))
        input_latent_parameters = torch.cat((initial_latent_states,
                                             zero_input_states),dim=1).T
        full_latent_parameters = self.encoder_step_2(input_latent_parameters)
        epsilons_across_indicies = []
        for i in range(self.num_latents):
            latent_mean_index = i*(self.covariance_rank + 1)
            latent_i_mean = full_latent_parameters[i*(self.covariance_rank + 1),:]
            latent_i_cov = sum([torch.outer(full_latent_parameters[latent_mean_index+j+1,:],full_latent_parameters[latent_mean_index+j+1,:]) for j in range(self.covariance_rank)])
            cholesky_i = torch.linalg.cholesky(latent_i_cov)
            epsilon_samples = torch.outer(latent_i_mean,torch.ones(self.epsilon_samples)) + cholesky_i @ torch.tensor(np.random.normal(0,1,(time_steps,self.epsilon_samples),dtype=torch.float))
            epsilons_across_indicies.append(epsilon_samples)
        
        return full_latent_parameters

sin_data_x,sin_data_y = imbedded_sin_bowl(0,8,100,1)
dataset = torch.stack([sin_data_x,sin_data_y]).T
model = BiLSTM(2).to(device)
output,_ = model(dataset)
#plt.scatter(sin_data_x,sin_data_y)
#plt.show()