from torch import nn , optim
import torch
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
        return means.reshape(batches,self.dim_latents,time_steps),sds_tensor**2

class LSTM_Posterior_VI2(nn.Module):
    def __init__(self,device,dim_latents,dim_observations,data_to_init_hidden_state_size=100,hidden_state_to_posterior_size=100):
        super().__init__()
        self.dim_latents = dim_latents
        self.dim_observations = dim_observations
        self.data_to_hidden_layer = nn.LSTM(dim_observations,data_to_init_hidden_state_size,batch_first=True)
        self.hidden_layer_to_initial = nn.Linear(data_to_init_hidden_state_size,hidden_state_to_posterior_size)
        self.initial_to_posterior_states = nn.LSTM(1,hidden_state_to_posterior_size,batch_first=True)
        self.to_mean_sd = nn.Linear(hidden_state_to_posterior_size,3*dim_latents)
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
        means,sds,lr = mean_sds[:,:,:self.dim_latents],mean_sds[:,:,self.dim_latents:2*self.dim_latents],mean_sds[:,:,2*self.dim_latents:]
        def outer_prods(matrix):
            return torch.stack([torch.outer(matrix[:,i],matrix[:,i]) for i in range(matrix.shape[1])])
        covs = torch.vmap(outer_prods)(lr)
        sds_tensor = torch.stack([torch.diag_embed(sds[batch_i,:,:].T) for batch_i in range(sds.shape[0])])
        #print(sds_tensor.shape)
        return means.reshape(batches,self.dim_latents,time_steps),covs+sds_tensor**2