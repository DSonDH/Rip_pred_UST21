
import torch.nn.functional as F
from torch import nn
import torch
import numpy as np


class DNN(nn.Module):
    def __init__(self, 
                 in_features,
                 out_features, 
                 input_len,
                 pred_len, 
                 dropout = 0.5,
                 positionalE = False, 
                ):
        super(DNN, self).__init__()

        # input (b, t, f, c)
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.LeakyReLU()

        self.fc2 = nn.Linear(out_features, out_features2)
        self.bn1 = nn.BatchNorm1d(out_features2)
        self.relu = nn.LeakyReLU()

        self.fc2 = nn.Linear(out_features2, out_features3)
        self.bn1 = nn.BatchNorm1d(out_features3)
        self.relu = nn.LeakyReLU()

        self.fc2 = nn.Linear(out_features3, pred_len)
        self.bn1 = nn.BatchNorm1d(pred_len)
        self.relu = nn.LeakyReLU()


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
        PE = self.get_position_encoding(x)

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)

        return out