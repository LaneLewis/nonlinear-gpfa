import torch
def gaussian_process_prior(data_times,meanFunc,KFunc):
    means = meanFunc(data_times)
    K = KFunc(data_times)
    gp_prior = torch.distributions.MultivariateNormal(means,K)
    return gp_prior

def kernal_SE(tau,signal_var,noise_var):
    def kernal(data_times):
        data_length = len(data_times)
        data_times_square = torch.outer(torch.square(data_times),torch.ones(data_length))
        K = signal_var * torch.exp(-1*(data_times_square + data_times_square.T - 2 * torch.outer(data_times,data_times)
                                        )/(2*tau**2))+noise_var*torch.eye(data_length)
        return K
    return kernal

def zero_mean():
    def zero(data_times):
        return torch.zeros(len(data_times),dtype=float)
    return zero