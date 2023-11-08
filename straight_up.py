
import torch
import copy
import torch.optim as optim
import torch.nn as nn
import tqdm as tqdm
from torch import nn , optim
from torch.utils.data import TensorDataset,DataLoader
#from helper_funcs.gpml_gen import GPML_Generating_Model
from helper_funcs.gm_ml_mv import GPML_MV_Generating_Model
from helper_funcs.lstm_posterior import LSTM_Posterior_VI,LSTM_Posterior_VI2
from helper_funcs.feedforward_nn import nn_embedding_model
from datasets.nn_embedding import nn_embedding_dataset,linear_embedding_dataset
from gaussian_process import kernal_SE,zero_mean
import matplotlib.pyplot as plt
from gaussian_process import kernal_SE

class GPML_ML_NN():
    def __init__(self,device,latent_dims,observed_dims,embedding_nn_model,sampling_number=1000):
        self.embedding_nn_model = embedding_nn_model
        self.taus = torch.tensor([10.0]*latent_dims,requires_grad=True,device=device)
        self.kernal_noise_sds = 0.01*torch.ones(latent_dims,device=device)
        self.kernal_signal_sds = 1.0 - self.kernal_noise_sds
        self.observation_noises = torch.tensor([0.1]*observed_dims,requires_grad=True,device=device)

        self.latent_dim = latent_dims
        self.observation_dims = observed_dims
        self.observation_cov = torch.diag(self.observation_noises)
        self.vectorized_embedding_func = torch.vmap(embedding_nn_model)
        #generating_model
        self.sampling_number = sampling_number
        self.train_loss_trajectory = []
        self.test_loss_trajectory = []

    def block_K_cov(self,times):
        Ks = []
        for i in range(self.latent_dim):
            Ks.append(kernal_SE(self.taus[i],self.kernal_signal_sds[i]**2,self.kernal_noise_sds[i]**2)(times))
        flat_z_block_cov = torch.block_diag(*Ks).float()
        return flat_z_block_cov
    
    def fit(self,X_train,X_time_train,X_test,X_time_test,epochs=1,learning_rate=0.01,batch_size=2):
        #self.embedding_optimizer = optim.Adam(self.embedding_nn_model.parameters(),lr=0.005)
        #self.optimizer_noise = optim.Adam(params=[self.observation_noises],lr=0.001)
        time_length_train = X_time_train.shape[0]
        time_length_test = X_time_test.shape[0]
        self.optimizer_taus = optim.Adam(params=[self.taus],lr=0.1)

        train_samples = torch.distributions.MultivariateNormal(torch.zeros((time_length_train*self.latent_dim)),torch.eye(time_length_train*self.latent_dim)).sample((self.sampling_number,))
        #test_samples = torch.distributions.MultivariateNormal(torch.zeros((time_length_test*self.latent_dim)),torch.eye(time_length_test*self.latent_dim)).sample((self.sampling_number,))
        train_samples = train_samples.reshape((self.sampling_number,time_length_train,self.latent_dim))
        #test_samples = test_samples.reshape((self.sampling_number,time_length_test,self.latent_dim))

        dataset = TensorDataset(X_train)
        batched_dataset = DataLoader(dataset,batch_size=batch_size)
        #epochs
        for epoch in range(epochs):
            #batches
            batch_count = 0
            batch_loss = 0.0

            for (batch_X,) in tqdm.tqdm(batched_dataset,desc=f"epoch: {epoch}",colour="cyan"):
                self.optimizer_taus.zero_grad()
                #self.embedding_optimizer.zero_grad()
                block_K_cov = self.block_K_cov(X_time_train)
                block_cholesky = torch.linalg.cholesky(block_K_cov)
                z_samples = torch.matmul(block_cholesky,train_samples)
                total_batch_loss = -1*self.batched_loss(batch_X,z_samples)
                total_batch_loss.backward()
                #self.embedding_optimizer.step()
                self.optimizer_taus.step()
                batch_count+=1
                batch_loss += total_batch_loss
            print(f"train av_batch_loss: {batch_loss/batch_count} | tau:{self.taus}| obs:{self.observation_noises}")
            print(block_cholesky)
            #print(f"test_loss:{self.batched_loss(X_test,X_test_times)}")
        return {"tau":self.taus,"kernal_signal_sd":self.kernal_signal_sds,"kernal_noise_sd":self.kernal_noise_sds}
    
    def batched_loss(self,batch_X,z_samples):
        batch_size = batch_X.shape[0]
        total_batch_loss = torch.zeros((1))
        for sub_batch_index in range(batch_size):
            total_batch_loss += self.lb_loss(batch_X[sub_batch_index,:,:],z_samples) #+ 100*torch.norm(self.observation_noises,2.0)
        return total_batch_loss
            
    def lb_loss(self,X,z_samples):
        '''Params:
            X: tensor of shape [time_dim,neurons]
            times: tensor of shape [time_dim]
            sample_size: int specifying how many times to sample from the vi dist in order to approx the elbo'''
        #begins the section for sampling the p(x|z) term
        #prepares some terms that will be used many times
        inv_observation_cov = torch.inverse(self.observation_cov)
        observation_log_det = torch.log(torch.det(self.observation_cov))
        #embeds the samples into the observation space mean. embedded_samples should have shape [samples, timelength, neurons]
        embedded_samples = self.vectorized_embedding_func(z_samples)
        # broadcast X to embedded mean_samples this should have shape [samples, timelength, neurons]
        samples_mean_subtracted_X = X - embedded_samples
        out = torch.linalg.norm(samples_mean_subtracted_X)
        #this should have shape [samples]
        log_likelihood_per_sample = batched_normal_dist_log_likelihood(samples_mean_subtracted_X,inv_observation_cov,observation_log_det)
        #final term
        expectation_term = torch.mean(log_likelihood_per_sample)
        return -1*out

def batched_normal_dist_log_likelihood(batch_mean_subtracted_X,inv_observation_cov,observation_log_det):
    #batch_mean_subtracted_X has dim [samples, timelength, neurons]
    def normal_log_likelihood_over_time(mean_subtracted_X):
        time_dim = mean_subtracted_X.shape[0]
        neuron_dim = mean_subtracted_X.shape[1]
        const_term = neuron_dim*torch.log(torch.tensor(2*torch.pi))
        #sums over all quadratic forms for t
        quadratic_term = torch.trace(torch.linalg.multi_dot([mean_subtracted_X,inv_observation_cov,mean_subtracted_X.T]))
        return -1/2*(time_dim*const_term + quadratic_term + time_dim*observation_log_det)
    return torch.vmap(normal_log_likelihood_over_time,in_dims=0,out_dims=0)(batch_mean_subtracted_X)

if torch.cuda.is_available(): 
 device = "cuda:0" 
else: 
 device = "cpu"

latent_dim = 1
observation_dim = 2
(X_train,X_times_train,z_train),(X_test,X_times_test,z_test),params = linear_embedding_dataset(device,train_num=100,test_num=1,time_divisions=200,latents=latent_dim,observed=observation_dim)
#embedding_nn = nn_embedding_model(latent_dim,observation_dim,non_output_layers=[(1,nn.Sigmoid())])
weights = torch.stack([(i+1)*torch.ones((latent_dim),requires_grad=False) for i in range(observation_dim)])
#print(X_train[0,:,:].shape)
#plt.plot(X_train[0,:,:])
#plt.savefig("./test.png")


def linear_embedding_model(weights):
    def marginalized(latent_values):
        lin_func = torch.matmul(weights,latent_values)
        return lin_func
    return marginalized
#latents = 1
embedding_func = torch.vmap(linear_embedding_model(weights))
with torch.no_grad():
    times = X_times_train[0,:]
    tau = 0.1
    signal = 0.999
    noise = 0.001
    obs_cov = 0.1*torch.diag(torch.ones(observation_dim))
    z_sample = torch.distributions.MultivariateNormal(torch.zeros((times.shape[0])),kernal_SE(tau,signal,noise)(times)).rsample((1,))
    embedded_mean = embedding_func(z_sample.T)
    obs_dist = torch.distributions.MultivariateNormal(torch.zeros(observation_dim),obs_cov)
    obs_sample = obs_dist.sample((times.shape[0],))
    obs_data = embedded_mean + obs_sample
    plt.plot(obs_data[:,0].detach().numpy())
    plt.plot(obs_data[:,1].detach().numpy())
    plt.savefig("./test.png")

double_batched_embedding = torch.vmap(embedding_func)
#L = torch.linalg.cholesky(kernal_SE(tau,signal,noise)(times))
#standard_sample = torch.distributions.MultivariateNormal(torch.zeros((times.shape[0])),torch.eye(times.shape[0])).rsample((1,))
#new_sample = torch.matmul(L,standard_sample.T)

samples = 1000
model_tau = torch.tensor([0.5]*latent_dim,requires_grad=True)
optimizer = optim.SGD([model_tau],lr=0.1)
for i in range(100):
    optimizer.zero_grad()
    z_model_sample = torch.distributions.MultivariateNormal(torch.zeros((times.shape[0])),kernal_SE(model_tau,signal,noise)(times)).rsample((samples,1,)).reshape((samples,times.shape[0],1))
    out_embedding = double_batched_embedding(z_model_sample)
    loss = torch.norm(obs_data - out_embedding)
    loss.backward()
    optimizer.step()
    print(model_tau)
plt.cla() 
plt.plot(out_embedding[0,:,0].detach().numpy())
plt.plot(out_embedding[0,:,1].detach().numpy())
plt.savefig("./test1.png")
#obs_sample = obs_dist.sample((times.shape[0],))
#obs_data = embedded_mean + obs_sample
#plt.plot(obs_data[:,0].detach().numpy())
#plt.plot(obs_data[:,1].detach().numpy())
#plt.savefig("./test.png")
#GPML_ML_NN(device,latent_dim,observation_dim,embedding_func,10000).fit(X_train,X_times_train[0,:],X_test,X_times_test[0,:],epochs=200,batch_size=5)
