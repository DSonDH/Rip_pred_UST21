from collections import OrderedDict
import torch.nn.functional as F
from torch import nn
import torch


class DNN(nn.Module):
    def __init__(self, 
                 features: list, 
                 pred_len: int, 
                 dropout = 0.5,
                 positionalE = False, 
                ):
        super(DNN, self).__init__()

        # FIXME: 
        # Batchnorm or LayerNorm
        # dropout or no dropout
        # feature variable 한개의 list 변수로 제어 ? 
        
        # input (b, t, f, c)
        layers = OrderedDict()
        for idx, params in enumerate(features):
            layer_name = f'linear_{idx + 1}'
            layers[layer_name] = nn.Linear(*params)
            activation_name = f'LeakyReLU_{idx + 1}'
            layers[activation_name] = nn.LeakyReLU()
            
        self.layers = nn.Sequential(layers)
        last_dim = features[-1][-1]
        self.output = nn.Linear(last_dim, pred_len)
        

    def get_position_encoding(self, x):
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype=torch.float32, device=x.device)
        temp1 = position.unsqueeze(1)  # 5 1
        temp2 = self.inv_timescales.unsqueeze(0)  # 1 256
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)  # 5 256
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)  #[T, C]
        signal = F.pad(signal, (0, 0, 0, self.pe_hidden_size % 2))
        signal = signal.view(1, max_length, self.pe_hidden_size)
    
        return signal
    
    def forward(self, x):
        # PE = self.get_position_encoding(x)

        # x = x.reshape((-1, x.shape[1] * x.shape[2]))

        hidden = self.layers(x)
        out = self.output(hidden)
        
        return out