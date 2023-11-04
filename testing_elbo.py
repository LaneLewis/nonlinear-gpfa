
import torch
import copy
import torch.optim as optim
import torch.nn as nn
import tqdm as tqdm
from torch.utils.data import TensorDataset,DataLoader
from helper_funcs.model_save import save_model,load_model
#from helper_funcs.gpml_gen import GPML_Generating_Model
from helper_funcs.gm_ml_mv import GPML_MV_Generating_Model
from helper_funcs.feedforward_nn import nn_embedding_model
from datasets.nn_embedding import nn_embedding_dataset

class GPML_VAE_NN():
    def __init__(self,device,latent_dims,observed_dims,embedding_nn_layers=[(10,nn.ReLU()),(10,nn.ReLU())],samples=500):
        self.nn_model = nn_embedding_model(latent_dims,observed_dims,embedding_nn_layers)
        self.taus = torch.rand(latent_dims,requires_grad=True,device=device,dtype=float)
        self.kernal_noise_sds = 0.01*torch.ones(latent_dims,device=device,dtype=float) #torch.rand(latent_dims,requires_grad=True,device=device)
        self.kernal_signal_sds = 1.0 - self.kernal_noise_sds #torch.rand(latent_dims,requires_grad=True,device=device)
        self.observation_noises = torch.rand(observed_dims,requires_grad=True,device=device,dtype=float)
        self.likelihood_model = GPML_MV_Generating_Model(self.nn_model.forward,self.taus,self.kernal_signal_sds,self.kernal_noise_sds,self.observation_noises)
        #generating_model
        self.samples = samples
        self.train_loss_trajectory = []
        self.test_loss_trajectory = []

    def fit(self,X_train,X_times_train,epochs=1,learning_rate=0.005,batch_size=2):

        self.optimizer_nn = optim.Adam(self.nn_model.parameters(),lr=learning_rate)
        self.optimizer_non_nn = optim.Adam(params=[self.taus,self.kernal_signal_sds,self.kernal_noise_sds,self.observation_noises],lr=learning_rate)
        dataset = TensorDataset(X_train,X_times_train)
        batched_dataset = DataLoader(dataset,batch_size=batch_size)
        #epochs
        for epoch in range(epochs):
            #batches
            for batch_X,batch_time in tqdm.tqdm(batched_dataset,desc=f"epoch: {epoch}",colour="cyan"):
                batch_size = batch_X.shape[0]
                time_dim = batch_time.shape[1]
                num_latents = self.likelihood_model.latent_dim
                #copies network and switches off the gradient
                #generating_model
                vi_means = [torch.ones(time_dim,dtype=float)]*num_latents
                vi_diag_covs = [2*torch.diag(torch.ones(time_dim,dtype=float))]*num_latents

                def batched_elbo(single_X,single_time):
                    return self.likelihood_model.approx_elbo_loss(vi_means,vi_diag_covs,single_X,single_time)
                
                batched_loss = torch.vmap(batched_elbo,randomness="same")
                batched_loss(batch_X,batch_time)
                #for sub_batch_index in range(batch_size):
                   
                    #out = self.likelihood_model.approx_elbo_loss(vi_means,vi_diag_covs,batch_X[sub_batch_index,:,:],batch_time[sub_batch_index,:])
        return {"tau":self.taus,"kernal_signal_sd":self.kernal_signal_sds,"kernal_noise_sd":self.kernal_noise_sds}
    
if torch.cuda.is_available(): 
 device = "cuda:0" 
else: 
 device = "cpu"

(X_train,X_times_train,z_train),(X_test,X_times_test,z_test),params = nn_embedding_dataset(device,train_num=4,test_num=1,time_divisions=300)
GPML_VAE_NN(device,1,2).fit(X_train,X_times_train)