import torch
from helper_funcs.feedforward_nn import nn_embedding_model
from helper_funcs.gm_ml_mv import GPML_MV_Generating_Model
from sampling import Sampling
from torch import nn

def nn_embedding_dataset(device,train_num=100,test_num=15,time_divisions=300,latents=2,observed=3):
    times = torch.linspace(0.0,10.0,time_divisions,device=device).float()
    #initial parameters
    kernal_noise_sds = 0.01*torch.ones(latents,device=device) #torch.rand(latent_dims,requires_grad=True,device=device)
    kernal_signal_sds = 1.0 - kernal_noise_sds
    tau = torch.tensor([0.01]*latents,device=device).float()
    observation_noise = torch.tensor([0.1]*observed,device=device).float()
    nn_model = nn_embedding_model(latents,observed,non_output_layers=[(1,nn.Sigmoid())])
    embedding_func = nn_model.forward
    dataset_generator = GPML_MV_Generating_Model(embedding_func,tau,kernal_signal_sds,kernal_noise_sds,observation_noise)

    Xs = []
    zs = []
    for _ in range(train_num+test_num):
        with torch.no_grad():
            X,z = dataset_generator.sample_joint(times)
            Xs.append(X)
            zs.append(z)

    X_out = torch.stack(Xs,dim=0)
    z_out = torch.stack(zs,dim=0)
    times_out = torch.stack((train_num+test_num)*[times])
    params = {"taus":tau,"kernal_signal_sds":kernal_signal_sds,"kernal_noise_sds":kernal_noise_sds,"observation_noise":observation_noise}
    train_X,train_t,train_z = X_out[0:train_num,:,:],times_out[0:train_num,:],z_out[0:train_num,:,:]
    test_X,test_t,test_z = X_out[train_num:,:,:],times_out[train_num:,:],z_out[train_num:,:,:]
    return (train_X,train_t,train_z),(test_X,test_t,test_z),params

def linear_embedding_dataset(device,train_num=100,test_num=15,time_divisions=300,latents=2,observed=3):
    times = torch.linspace(0.0,1.0,time_divisions,device=device).float()
    #initial parameters
    kernal_noise_sds = 0.01*torch.ones(latents,device=device) #torch.rand(latent_dims,requires_grad=True,device=device)
    kernal_signal_sds = 1.0 - kernal_noise_sds
    taus = torch.tensor([1000000.0]*latents,device=device).float()
    observation_noises = torch.tensor([0.1]*observed,device=device).float()
    weights = torch.ones((observed,latents))
    def linear_embedding_model(weights):
        def marginalized(latent_values):
            return torch.matmul(weights,latent_values)
        return marginalized
    
    embedding_func = torch.vmap(linear_embedding_model(weights))
    dataset_generator = Sampling(embedding_func,taus,observation_noises,kernal_noise_sds,kernal_signal_sds)
    with torch.no_grad():
        train_X,train_Z = dataset_generator.sample_joint(train_num,times)
        test_X,test_Z = dataset_generator.sample_joint(test_num,times)
    train_t = torch.stack([times]*train_num)
    test_t = torch.stack([times]*test_num)

    params = {"taus":taus,"kernal_signal_sds":kernal_signal_sds,"kernal_noise_sds":kernal_noise_sds,"observation_noise":observation_noises}
    return (train_X,train_t,train_Z),(test_X,test_t,test_Z),params