import torch
import copy
import torch.optim as optim
import torch.nn as nn
import tqdm as tqdm
from torch.utils.data import TensorDataset,DataLoader
#from helper_funcs.gpml_gen import GPML_Generating_Model
from helper_funcs.gm_ml_mv import GPML_MV_Generating_Model
from helper_funcs.feedforward_nn import nn_embedding_model
from datasets.nn_embedding import nn_embedding_dataset

class GPML_MCMC_NN():
    def __init__(self,device,latent_dims,observed_dims,nn_layers=[(10,nn.ReLU()),(10,nn.ReLU())],posterior_samples=200,burn_in=20,gradients_per_expectation=5):
        self.nn_model = nn_embedding_model(latent_dims,observed_dims,nn_layers)
        #for param in self.nn_model.parameters():
        #    param.requires_grad=False
        self.taus = torch.rand(latent_dims,requires_grad=True,device=device)
        self.kernal_signal_sds = torch.rand(latent_dims,requires_grad=True,device=device)
        self.kernal_noise_sds = torch.rand(latent_dims,requires_grad=True,device=device)
        self.observation_noises = torch.rand(observed_dims,requires_grad=True,device=device)
        self.posterior_samples = posterior_samples
        self.burn_in= burn_in
        self.gradients_per_expectation = gradients_per_expectation
        self.likelihood_model = GPML_MV_Generating_Model(self.nn_model.forward,self.taus,self.kernal_signal_sds,self.kernal_noise_sds,self.observation_noises)
        #copies network and switches off the gradient
        nn_frozen_copy = self.nn_model.copy_and_freeze()
        #generating_model
        self.generating_model = GPML_MV_Generating_Model(nn_frozen_copy.forward,self.taus.detach().clone(),
                                    self.kernal_signal_sds.detach().clone(),self.kernal_noise_sds.detach().clone(),
                                    self.observation_noises.detach().clone())
        
    def fit(self,X_train,X_times_train,X_test,X_times_test,epochs=1000,learning_rate=0.005,batch_size=2):
        self.optimizer_nn = optim.Adam(self.nn_model.parameters(),lr=learning_rate)
        self.optimizer_non_nn = optim.Adam(params=[self.taus,self.kernal_signal_sds,self.kernal_noise_sds,self.observation_noises],lr=learning_rate)
        dataset = TensorDataset(X_train,X_times_train)
        batched_dataset = DataLoader(dataset,batch_size=batch_size)
        #epochs
        for epoch in range(epochs):
            #batches
            for batch_X,batch_time in tqdm.tqdm(batched_dataset,desc=f"epoch: {epoch+1}",colour="cyan"):
                batch_size = batch_X.shape[0]
                #expectation step
                self.optimizer_non_nn.zero_grad()
                self.optimizer_nn.zero_grad()
                #likelihood portion
                self.likelihood_model = GPML_MV_Generating_Model(self.nn_model.forward,self.taus,self.kernal_signal_sds,self.kernal_noise_sds,self.observation_noises)#simple_embed_wrapper(self.theta,self.tau,self.kernal_signal_sd,self.kernal_noise_sd,self.observation_noise)
                #copies network and switches off the gradient
                nn_frozen_copy = self.nn_model.copy_and_freeze()
                #generating_model
                self.generating_model = GPML_MV_Generating_Model(nn_frozen_copy.forward,self.taus.detach().clone(),
                                    self.kernal_signal_sds.detach().clone(),self.kernal_noise_sds.detach().clone(),
                                    self.observation_noises.detach().clone())
                #maximization step
                for _ in range(self.gradients_per_expectation):
                    batch_loss = self.batch_loss(batch_X,batch_time,batch_size)
                    batch_loss.backward()
                    self.optimizer_nn.step()
                    self.optimizer_non_nn.step()
            print(f"epoch={epoch}/{epochs} | train_loss: {self.dataset_loss(X_train,X_times_train)} | test_loss {self.dataset_loss(X_test,X_times_test)}")
        return {"tau":self.taus,"kernal_signal_sd":self.kernal_signal_sds,"kernal_noise_sd":self.kernal_noise_sds}
    
    def batch_loss(self,batch_X,batch_time,batch_size):
        batch_loss = torch.zeros(1,dtype=float)
        for sub_batch_index in range(batch_size):
            batch_loss += self.single_loss(batch_X[sub_batch_index],batch_time[sub_batch_index])
        batch_loss = batch_loss/batch_size
        return batch_loss
    
    def single_loss(self,X,times):
        likelihood_z_func = self.likelihood_model.joint_log_likelihood_given_z(X,times)
        approx_expectation = self.generating_model.expectation_of_z_given_X(X,times,likelihood_z_func,samples=self.posterior_samples,burn=self.burn_in)
        loss = -1*approx_expectation
        return loss
    
    def dataset_loss(self,Xs,times):
        return self.batch_loss(Xs,times,Xs.shape[0]).data[0]

    def show_fit(self,params):
        fitted_params = {"taus":self.taus,"kernal_signal_sds":self.kernal_signal_sds,"kernal_noise_sds":self.kernal_noise_sds}
        print(f"tau | true={params['taus'].data},fit={fitted_params['taus'].data} | diff = {params['taus'].data-fitted_params['taus'].data}")
        print(f"kernal_signal_sd | true={params['kernal_signal_sds'].data},fit={fitted_params['kernal_signal_sds'].data} | diff = {params['kernal_signal_sds'].data-fitted_params['kernal_signal_sds'].data}")
        print(f"kernal_noise_sd | true={params['kernal_noise_sds'].data},fit={fitted_params['kernal_noise_sds'].data} |  diff = {params['kernal_noise_sds'].data-fitted_params['kernal_noise_sds'].data}")

if torch.cuda.is_available(): 
 device = "cuda:0" 
else: 
 device = "cpu"

(X_train,X_times_train,z_train),(X_test,X_times_test,z_test),params = nn_embedding_dataset(device,train_num=4,test_num=1)
dataset = TensorDataset(X_train,X_times_train)
model = GPML_MCMC_NN(device,1,2)
print(f"initial | train_loss: {model.dataset_loss(X_train,X_times_train)} | test_loss {model.dataset_loss(X_test,X_times_test)}")
model.fit(X_train,X_times_train,X_test,X_times_test,epochs=30,batch_size=2,learning_rate=0.01)
model.show_fit(params)


#likelihood_model = GPML_MCMC(embedding_func,tau,kernal_signal_sd,kernal_noise_sd,observation_noise)

#copied_model = GPML_MCMC()
#print(gpml_model.sample_joint(times))
#print(gpml_model.mcmc_sample_posterior(X,times))

