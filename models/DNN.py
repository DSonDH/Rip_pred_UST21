from collections import OrderedDict
from torch import nn


class DNN(nn.Module):
    """ simple dnn
    It only treats depth, hidden usit size, dropout rate.
    """
    def __init__(self, 
                 features: list, 
                 pred_len: int, 
                 dropout:float = 0.5,
                 positionalE:bool = False, 
                ):
        super(DNN, self).__init__()

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
        self.relu = nn.ReLU()


    def forward(self, x):
        # PE = self.get_position_encoding(x)
        # x = x.reshape((-1, x.shape[1] * x.shape[2]))

        hidden = self.layers(x)
        out = self.output(hidden)
        out = self.relu(out)
        
        return out