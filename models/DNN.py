from collections import OrderedDict
import torch.nn.functional as F
from torch import nn
import torch


class DNN(nn.Module):
    def __init__(self, 
                 features: list, 
                 pred_len: int, 
                 dropout:float = 0.5,
                 positionalE:bool = False, 
                ):
        super(DNN, self).__init__()

        # FIXME: 
        # Batchnorm or LayerNorm
        # dropout or no dropout
        
        layers = OrderedDict()
        for idx, params in enumerate(features):
            layer_name = f'linear_{idx + 1}'
            layers[layer_name] = nn.Linear(*params)
            
            activation_name = f'LeakyReLU_{idx + 1}'
            layers[activation_name] = nn.LeakyReLU()
            
            dropout_name = f'dropOut_{idx + 1}'
            layers[dropout_name] = nn.Dropout(dropout)
            
        self.layers = nn.Sequential(layers)
        last_dim = features[-1][-1]
        self.output = nn.Linear(last_dim, pred_len)

        #TODO: dropout scaling

    def forward(self, x):
        # PE = self.get_position_encoding(x)
        # x = x.reshape((-1, x.shape[1] * x.shape[2]))
        hidden = self.layers(x)
        out = self.output(hidden)
        
        return out