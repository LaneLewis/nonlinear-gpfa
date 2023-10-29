import torch
from torch import nn , optim
from torch.nn.modules import Module
from helper_funcs.feedforward_nn import FeedforwardNN,nn_embedding_func

input_dims = 1
observation_dims = 2
embedding_func,model_params = nn_embedding_func(input_dims,observation_dims)
optimizer = optim.Adam(model_params,0.1)

fake_input = torch.ones(3,dtype=float).reshape((3,1))
print(fake_input.dtype)
nn_output = torch.vmap(embedding_func)(fake_input)
print(nn_output)
target_output = torch.ones(3,2)
#loss = torch.sum(target_output - nn_output)
#loss.backward()
#optimizer.step()