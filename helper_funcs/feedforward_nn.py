import torch
import copy
from torch import nn , optim
from torch.nn.modules import Module

class FeedforwardNN(nn.Module):
    def __init__(self,layers_list,input_size,device="cpu"):
        super().__init__()
        self.layers = nn.ModuleList()
        self.input_dim = input_size
        for size,activation in layers_list:
            linear_layer = nn.Linear(input_size,size)
            self.layers.append(linear_layer)
            input_size = size
            if activation is not None:
                assert isinstance(activation, Module),"Each tuples should contain a size (int) and a torch.nn.modules.Module."
                self.layers.append(activation)
        self.to(device)

    def forward(self,input_data):
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data
    
    def copy_and_freeze(self):
        nn_copy = copy.deepcopy(self)
        for param in nn_copy.parameters():
            param.requires_grad = False
        return nn_copy

def nn_embedding_model(input_dim,output_dim,non_output_layers=[(10,nn.ReLU()),(10,nn.ReLU())]):
    nn_layers = [*non_output_layers,(output_dim,None)]
    nn_model = FeedforwardNN(nn_layers,input_dim)
    return nn_model