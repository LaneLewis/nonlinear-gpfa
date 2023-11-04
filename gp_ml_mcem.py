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

class GPML_MCMC_NN():
    def __init__(self,device,latent_dims,observed_dims,nn_layers=[(10,nn.ReLU()),(10,nn.ReLU())]
                 ,posterior_samples=500,burn_in=20,gradients_per_expectation=5,epoch_samples=300):
        self.nn_model = nn_embedding_model(latent_dims,observed_dims,nn_layers)
        self.taus = torch.rand(latent_dims,requires_grad=True,device=device)
        self.kernal_noise_sds = torch.tensor(0.001,device=device) #torch.rand(latent_dims,requires_grad=True,device=device)
        self.kernal_signal_sds = torch.tensor(1.0 - self.kernal_noise_sds,device=device) #torch.rand(latent_dims,requires_grad=True,device=device)
        self.observation_noises = torch.rand(observed_dims,requires_grad=True,device=device)
        self.posterior_samples = posterior_samples
        self.burn_in= burn_in
        self.epoch_samples = epoch_samples
        self.gradients_per_expectation = gradients_per_expectation
        self.likelihood_model = GPML_MV_Generating_Model(self.nn_model.forward,self.taus,self.kernal_signal_sds,self.kernal_noise_sds,self.observation_noises)
        #copies network and switches off the gradient
        nn_frozen_copy = self.nn_model.copy_and_freeze()
        #generating_model
        self.generating_model = GPML_MV_Generating_Model(nn_frozen_copy.forward,self.taus.detach().clone(),
                                    self.kernal_signal_sds.detach().clone(),self.kernal_noise_sds.detach().clone(),
                                    self.observation_noises.detach().clone())
        self.train_loss_trajectory = []
        self.test_loss_trajectory = []

    def fit(self,X_train,X_times_train,X_test,X_times_test,epochs=1000,learning_rate=0.005,batch_size=2):
        self.optimizer_nn = optim.Adam(self.nn_model.parameters(),lr=learning_rate)
        self.optimizer_non_nn = optim.Adam(params=[self.taus,self.kernal_signal_sds,self.kernal_noise_sds,self.observation_noises],lr=learning_rate)
        dataset = TensorDataset(X_train,X_times_train)
        batched_dataset = DataLoader(dataset,batch_size=batch_size)
        #epochs
        for epoch in range(epochs):
            #batches
            for batch_X,batch_time in tqdm.tqdm(batched_dataset,desc=f"epoch: {epoch}",colour="cyan"):
                batch_size = batch_X.shape[0]
                #expectation step
                self.optimizer_non_nn.zero_grad()
                self.optimizer_nn.zero_grad()
                #copies network and switches off the gradient
                nn_frozen_copy = self.nn_model.copy_and_freeze()
                #generating_model
                self.generating_model = GPML_MV_Generating_Model(nn_frozen_copy.forward,self.taus.detach().clone(),
                                    self.kernal_signal_sds.detach().clone(),self.kernal_noise_sds.detach().clone(),
                                    self.observation_noises.detach().clone())
                
                ####Expectation Step###
                self.vec_likelihood_log_X_given_z = torch.vmap(self.likelihood_model.joint_log_likelihood_given_z)
                #generates samples of z and the likelihood function to apply over the samples.
                batch_samples = []
                likelihood_z_funcs = []
                for sub_batch_index in range(batch_size):
                    z_samples = self.generating_model.mcmc_sample_posterior(batch_X[sub_batch_index],batch_time[sub_batch_index],samples=self.posterior_samples)
                    z_samples_corr_shape = z_samples.reshape((self.posterior_samples,batch_time[sub_batch_index].shape[0],1))
                    batch_samples.append(z_samples_corr_shape)
                    likelihood_z_func = torch.vmap(self.likelihood_model.joint_log_likelihood_given_z(batch_X[sub_batch_index],batch_time[sub_batch_index]))
                    likelihood_z_funcs.append(likelihood_z_func)
                #maximizes the objective using a gradients a number of times
                for _ in range(self.gradients_per_expectation):
                    total_batch_ll = torch.zeros(1,dtype=float)
                    for sub_batch_index in range(batch_size):
                        likelihood_of_each_z = likelihood_z_funcs[sub_batch_index](batch_samples[sub_batch_index])
                        ###this value is the approximated log likelihood that we want to maximize###
                        approx_expected_ll = torch.mean(likelihood_of_each_z)
                        total_batch_ll += approx_expected_ll
                    total_batch_ll = total_batch_ll/batch_size
                    batch_loss = -1*total_batch_ll
                    batch_loss.backward()
                    self.optimizer_nn.step()
                    self.optimizer_non_nn.step()
                    self.optimizer_non_nn.zero_grad()
                    self.optimizer_nn.zero_grad()
            
            epoch_train_loss = self.dataset_loss(X_train,X_times_train,posterior_samples=self.epoch_samples)
            epoch_test_loss = self.dataset_loss(X_test,X_times_test)
            self.train_loss_trajectory.append(epoch_train_loss)
            self.test_loss_trajectory.append(epoch_test_loss)
            print(f"epoch={epoch}/{epochs} | train_loss: {epoch_train_loss} | test_loss {epoch_test_loss}")
            print(self.taus.data)
        return {"tau":self.taus,"kernal_signal_sd":self.kernal_signal_sds,"kernal_noise_sd":self.kernal_noise_sds}
    
    def batch_loss(self,batch_X,batch_time,batch_size,posterior_samples=200):
        batch_loss = torch.zeros(1,dtype=float)
        for sub_batch_index in range(batch_size):
            batch_loss += self.single_loss(batch_X[sub_batch_index],batch_time[sub_batch_index],posterior_samples=posterior_samples)
        batch_loss = batch_loss/batch_size
        return batch_loss
    
    def single_loss(self,X,times,posterior_samples=200):
        likelihood_z_func = self.likelihood_model.joint_log_likelihood_given_z(X,times)
        approx_expectation = self.generating_model.expectation_of_z_given_X(X,times,likelihood_z_func,samples=posterior_samples,burn=self.burn_in)
        loss = -1*approx_expectation
        return loss
    
    def dataset_loss(self,Xs,times,posterior_samples=200):
        return self.batch_loss(Xs,times,Xs.shape[0],posterior_samples=posterior_samples).data[0]

    def show_fit(self,params):
        fitted_params = {"taus":self.taus,"kernal_signal_sds":self.kernal_signal_sds,"kernal_noise_sds":self.kernal_noise_sds}
        print(f"tau | true={params['taus'].data},fit={fitted_params['taus'].data} | diff = {params['taus'].data-fitted_params['taus'].data}")
        print(f"kernal_signal_sd | true={params['kernal_signal_sds'].data},fit={fitted_params['kernal_signal_sds'].data} | diff = {params['kernal_signal_sds'].data-fitted_params['kernal_signal_sds'].data}")
        print(f"kernal_noise_sd | true={params['kernal_noise_sds'].data},fit={fitted_params['kernal_noise_sds'].data} |  diff = {params['kernal_noise_sds'].data-fitted_params['kernal_noise_sds'].data}")

if torch.cuda.is_available(): 
 device = "cuda:0" 
else: 
 device = "cpu"

(X_train,X_times_train,z_train),(X_test,X_times_test,z_test),params = nn_embedding_dataset(device,train_num=4,test_num=1,time_divisions=300)
dataset = TensorDataset(X_train,X_times_train)
print("dataset built")
model = GPML_MCMC_NN(device,1,2,posterior_samples=500,burn_in=50,gradients_per_expectation=10,epoch_samples=200)
save_model("untrained",model)
print(f"initial | train_loss: {model.dataset_loss(X_train,X_times_train)} | test_loss {model.dataset_loss(X_test,X_times_test)}")
model.fit(X_train,X_times_train,X_test,X_times_test,epochs=40,batch_size=2,learning_rate=0.005)
model.show_fit(params)
save_model("trained",model)