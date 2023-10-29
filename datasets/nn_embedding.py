import torch
from helper_funcs.feedforward_nn import nn_embedding_model
from helper_funcs.gm_ml_mv import GPML_MV_Generating_Model

def nn_embedding_dataset(device,train_num=100,test_num=15):
    times = torch.linspace(0.0,1.0,300,dtype=float,device=device)
    #initial parameters
    tau = torch.tensor([0.5],dtype=float,device=device)
    kernal_signal_sd = torch.tensor([0.1],dtype=float,device=device)
    kernal_noise_sd = torch.tensor([0.1],dtype=float,device=device)
    observation_noise = torch.tensor([0.1,0.1],dtype=float,device=device)
    nn_model = nn_embedding_model(1,2)
    embedding_func = nn_model.forward
    dataset_generator = GPML_MV_Generating_Model(embedding_func,tau,kernal_signal_sd,kernal_noise_sd,observation_noise)
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
    params = {"taus":tau,"kernal_signal_sds":kernal_signal_sd,"kernal_noise_sds":kernal_noise_sd}
    train_X,train_t,train_z = X_out[0:train_num,:,:],times_out[0:train_num,:],z_out[0:train_num,:,:]
    test_X,test_t,test_z = X_out[train_num:,:,:],times_out[train_num:,:],z_out[train_num:,:,:]
    return (train_X,train_t,train_z),(test_X,test_t,test_z),params
