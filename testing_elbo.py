
import torch
import copy
import torch.optim as optim
import torch.nn as nn
import tqdm as tqdm
from torch import nn , optim
from torch.nn.modules import Module
from torch.utils.data import TensorDataset,DataLoader
from helper_funcs.model_save import save_model,load_model
#from helper_funcs.gpml_gen import GPML_Generating_Model
from helper_funcs.gm_ml_mv import GPML_MV_Generating_Model
from helper_funcs.feedforward_nn import nn_embedding_model
from datasets.nn_embedding import nn_embedding_dataset

class GPML_VAE_NN():
    def __init__(self,device,latent_dims,observed_dims,posterior_vi_nn_model,embedding_nn_model,vi_samples=500):
        self.embedding_nn_model = embedding_nn_model
        self.posterior_vi_nn_model = posterior_vi_nn_model
        self.taus = torch.rand(latent_dims,requires_grad=True,device=device)
        self.kernal_noise_sds = 0.01*torch.ones(latent_dims,device=device) #torch.rand(latent_dims,requires_grad=True,device=device)
        self.kernal_signal_sds = 1.0 - self.kernal_noise_sds #torch.rand(latent_dims,requires_grad=True,device=device)
        self.observation_noises = torch.rand(observed_dims,requires_grad=True,device=device)
        self.likelihood_model = GPML_MV_Generating_Model(self.embedding_nn_model.forward,self.taus,self.kernal_signal_sds,self.kernal_noise_sds,self.observation_noises)
        #generating_model
        self.vi_samples = vi_samples
        self.train_loss_trajectory = []
        self.test_loss_trajectory = []

    def fit(self,X_train,X_times_train,epochs=1,learning_rate=0.001,batch_size=2):
        self.embedding_optimizer = optim.Adam(self.embedding_nn_model.parameters(),lr=learning_rate)
        self.posterior_optimizer = optim.Adam(self.posterior_vi_nn_model.parameters(),lr=learning_rate)
        self.optimizer_non_nn = optim.Adam(params=[self.taus,self.kernal_signal_sds,self.kernal_noise_sds,self.observation_noises],lr=learning_rate)
        dataset = TensorDataset(X_train,X_times_train)
        batched_dataset = DataLoader(dataset,batch_size=batch_size)
        #epochs
        for epoch in range(epochs):
            #batches
            for batch_X,batch_time in tqdm.tqdm(batched_dataset,desc=f"epoch: {epoch}",colour="cyan"):
                self.embedding_optimizer.zero_grad()
                self.posterior_optimizer.zero_grad()
                self.optimizer_non_nn.zero_grad()
                batch_size = batch_X.shape[0]
                vi_means,vi_covs = self.posterior_vi_nn_model(batch_X)
                total_batch_loss = torch.zeros((1))
                for sub_batch_index in range(batch_size):
                    indiv_loss = -1*self.likelihood_model.approx_elbo_loss(vi_means[sub_batch_index,:,:],vi_covs[sub_batch_index,:,:],batch_X[sub_batch_index,:,:],batch_time[sub_batch_index,:])
                    total_batch_loss += indiv_loss
                total_batch_loss.backward()
                self.embedding_optimizer.step()
                self.posterior_optimizer.step()
                self.optimizer_non_nn.step()
            print(f"total_batch_loss: {total_batch_loss} | tau:{self.taus}")
        return {"tau":self.taus,"kernal_signal_sd":self.kernal_signal_sds,"kernal_noise_sd":self.kernal_noise_sds}
    
if torch.cuda.is_available(): 
 device = "cuda:0" 
else: 
 device = "cpu"

class LSTM_Posterior_VI(nn.Module):
    def __init__(self,device,dim_latents,dim_observations,data_to_init_hidden_state_size=100,hidden_state_to_posterior_size=100):
        super().__init__()
        self.dim_latents = dim_latents
        self.dim_observations = dim_observations
        self.data_to_hidden_layer = nn.LSTM(dim_observations,data_to_init_hidden_state_size,batch_first=True)
        self.hidden_layer_to_initial = nn.Linear(data_to_init_hidden_state_size,hidden_state_to_posterior_size)
        self.initial_to_posterior_states = nn.LSTM(1,hidden_state_to_posterior_size,batch_first=True)
        self.to_mean_sd = nn.Linear(hidden_state_to_posterior_size,2*dim_latents)
        self.device = device
    def forward(self,X):
        #assmues X has shape [batches,time,neurons]
        batches = X.shape[0]
        time_steps = X.shape[1]
        observed = X.shape[2]
        X = X.float()
        #should have shape [hidden_dim_1]
        hidden_states,(_,_) = self.data_to_hidden_layer(X)
        last_hidden_states = hidden_states[:,-1,:]
        dynamics_initial_cond = self.hidden_layer_to_initial(last_hidden_states)
        dummy_inputs = torch.zeros((batches,time_steps,1)).float()
        dummy_initial_cxs =  torch.zeros(dynamics_initial_cond.shape).float()
        dynamics_hidden_states,_ = self.initial_to_posterior_states(dummy_inputs,(dynamics_initial_cond.reshape(1,batches,dynamics_initial_cond.shape[1]),dummy_initial_cxs.reshape(1,batches,dummy_initial_cxs.shape[1])))
        mean_sds = self.to_mean_sd(dynamics_hidden_states)
        means,sds = mean_sds[:,:,:self.dim_latents],mean_sds[:,:,self.dim_latents:]
        sds_tensor = torch.stack([torch.diag_embed(sds[batch_i,:,:].T) for batch_i in range(sds.shape[0])])
        #sds is currently [batch,time,latents]
        #want [batch,latents,time,time]
        #print(torch.diag_embed(sds[0,:,:].T).shape)
        #print(torch.vmap(torch.diag_embed)(sds).shape)
        return means.reshape(batches,self.dim_latents,time_steps),sds_tensor**2
latent_dim = 1
observation_dim = 6
(X_train,X_times_train,z_train),(X_test,X_times_test,z_test),params = nn_embedding_dataset(device,train_num=50,test_num=1,time_divisions=300,latents=latent_dim,observed=observation_dim)
posterior_nn = LSTM_Posterior_VI(device,latent_dim,observation_dim,100,100)
embedding_nn = nn_embedding_model(latent_dim,observation_dim)
GPML_VAE_NN(device,latent_dim,observation_dim,posterior_nn,embedding_nn).fit(X_train,X_times_train,epochs=500,batch_size=5)