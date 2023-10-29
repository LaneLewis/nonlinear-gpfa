#note this doesn't work well because we are finding a direct likelihood and not a log likelihood.
import torch
from torch import nn , optim
from torch.nn.modules import Module
from gaussian_process import gaussian_process_prior,kernal_SE,zero_mean
from torchviz import make_dot
from tqdm import tqdm

class FeedforwardNN(nn.Module):
    def __init__(self,layers_list,input_size,learning_rate=0.01,optimizer=optim.Adam):
        super().__init__()
        self.layers = nn.ModuleList()
        self.input_dim = input_size
        for size,activation in layers_list:
            self.layers.append(nn.Linear(input_size,size))
            input_size = size
            if activation is not None:
                print(activation)
                assert isinstance(activation, Module),"Each tuples should contain a size (int) and a torch.nn.modules.Module."
                self.layers.append(activation)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.learning_rate = learning_rate
        self.optimizer = optimizer(params=self.parameters(), lr=learning_rate)

    def forward(self,input_data):
        for layer in self.layers:
            input_data = layer(input_data)
#ff_nn = FeedforwardNN([(10,nn.ReLU()),(10,nn.ReLU()),(observation_dims,nn.Linear())],1)

def normal_dist_likelihood(mean,covariance):
    dim = len(mean)
    constant = (2*torch.pi)**(-1*dim/2)*torch.det(covariance)**(-1/2)
    precision = torch.inverse(covariance)
    def likelihood(data_obs):
        exp_term = -1/2*torch.matmul(torch.matmul((data_obs - mean),precision),(data_obs - mean))
        return constant*torch.exp(exp_term)
    return likelihood

def simple_embedding(theta):
    def embed(zi):
        out_arr = torch.zeros(2)
        out_arr[0] = zi*theta
        out_arr[1] = zi**2*theta
        return out_arr
    return embed

def joint_normal_embedded_likelihood(z_vec,embedding_func,cov,data_obs):
    running_prod = 1
    for i,zi in enumerate(z_vec):
        mean_i = embedding_func(zi)
        data_i = data_obs[i]
        running_prod *= normal_dist_likelihood(mean_i,cov)(data_i)
    return running_prod

def simple_model(data,data_observation_times,epochs=1,sample_number=1,
                 theta_0=None,tau_0=None,kernal_signal_sd_0=None,kernal_noise_sd_0=None,observation_sds_0=None):
    param_dict = locals()
    #gp prior params
    if tau_0 != None:
        tau = torch.tensor(tau_0,requires_grad=True,dtype=float)
    else:
        tau = torch.rand(1,requires_grad=True,dtype=float)
    if kernal_signal_sd_0 != None:
        kernal_signal_sd = torch.tensor(kernal_signal_sd_0,requires_grad=True,dtype=float)
    else:
        kernal_signal_sd = torch.rand(1,requires_grad=True,dtype=float)

    if kernal_noise_sd_0 != None:
        kernal_noise_sd = torch.tensor(kernal_noise_sd_0,requires_grad=True,dtype=float)
    else:
        kernal_noise_sd = torch.rand(1,requires_grad=True,dtype=float)

    if theta_0 != None:
        theta = torch.tensor(theta,requires_grad=True,dtype=float)
    else:
        theta = torch.rand(1,requires_grad=True,dtype=float)

    if observation_sds_0 != None:
        observation_sds = torch.tensor(observation_sds,requires_grad=True,dtype=float)
    else:
        observation_sds = torch.rand(data.shape[1],requires_grad=True,dtype=float)
    
    input_vars = {"tau":str(tau.data),"kernal_signal_sd":str(kernal_signal_sd.data),"kernal_noise_sd":str(kernal_noise_sd.data),"observation_sds":str(observation_sds.data)}
    embedding_func = simple_embedding(theta)
    optimizer = optim.Adam([tau,kernal_signal_sd, kernal_noise_sd, theta ,observation_sds ], lr=0.05)
    loss_tensor = torch.zeros(epochs)
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        latent_dist = latent_model_dist(tau,kernal_signal_sd,kernal_noise_sd,data_observation_times)
        likelihood_x_given_z = embedding_model_likelihood(theta,observation_sds,embedding_func,data)        
        z_vecs = latent_dist.rsample((sample_number,))
        likelihood_values = torch.zeros(sample_number)
        for i in range(sample_number):
            likelihood_values[i] = likelihood_x_given_z(z_vecs[i,:])
        loss = -1*torch.mean(likelihood_values)
        loss.backward()
        optimizer.step()
        if epoch%100 == 0:
            print(f"Epoch: {epoch}/{epochs} | loss = {loss}")
        loss_tensor[epoch] = loss.data

    fit_vars = {"tau":str(tau.data),"kernal_signal_sd":str(kernal_signal_sd.data),"kernal_noise_sd":str(kernal_noise_sd.data),"observation_sds":str(observation_sds.data)}
    return input_vars,fit_vars

def latent_model_dist(tau,kernal_signal_sd,kernal_noise_sd,observation_times):
    K = kernal_SE(tau,kernal_signal_sd**2,kernal_noise_sd**2)
    m = zero_mean()
    z_dist= gaussian_process_prior(observation_times,m,K)
    return z_dist

def embedding_model_likelihood(theta,observation_sds,embedding_func,data):
    cov = torch.diag(observation_sds**2).float()
    embedding_func = simple_embedding(theta)
    def likelihood_z_func(z_vec):
        return joint_normal_embedded_likelihood(z_vec,embedding_func,cov,data)
    return likelihood_z_func

def simple_training_data(theta=0.5,tau=1.0,kernal_signal_sd=0.1,kernal_noise_sd=0.01,observation_sds=[1.0,0.5],observation_times=torch.linspace(0.0,1.0,10,dtype=float)):
    param_dict = locals()
    latent_dist = latent_model_dist(tau,kernal_signal_sd,kernal_noise_sd,observation_times)
    theta = torch.tensor(1.0).float()
    cov = torch.diag(torch.tensor(observation_sds)**2).float()
    embedding_func = simple_embedding(theta)
    #multi sample loop would start here
    z_vec = latent_dist.sample()
    data_point = []
    for z_i in z_vec:
        data_point_row_dist = torch.distributions.MultivariateNormal(embedding_func(z_i),cov)
        data_point_row = data_point_row_dist.sample()
        data_point.append(data_point_row)
    data = torch.stack(data_point)
    param_dict["z"] = z_vec
    return data,observation_times,z_vec,param_dict

#data,data_observation_times,true_params = simple_training_data()
#print(simple_model(data,data_observation_times,epochs=700,sample_number=100))
#print(true_params)
#dist1 = torch.distributions.MultivariateNormal(torch.tensor([1.0,0.0]),torch.diag(torch.tensor([1.0,2.0]))).rsample()

#simple_example()
#loss.backward()
#print(sigmas.grad)
#print(theta.grad)
#probability_Z = pZ(tau,kernal_signal_var,kernal_noise_var)
#probability_X_given_Z = pXgivenZ(ff_nn,observation_sigmas)

#Z_samples = probability_Z(observation_times).rsample((100,))

