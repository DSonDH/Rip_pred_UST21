from collections import OrderedDict
from torch import nn

class Simple1DCNN(nn.Module):
    """ simple 1dnn
    It only treats depth, hidden usit size, dropout rate.

    simpleDNN과의 차이점은 
    convolution 연산에 의해 local feature 추출이 용이한 것
    """

    def __init__(self,
                 features: list,
                 input_len: int,
                 pred_len: int,
                 kernelSize: int,
                 dilation: int,
                 dropout: float = 0.5,
                 isDepthWise: bool = False,
                 ):

        super(Simple1DCNN, self).__init__()
        self.features = features
        self.input_len = input_len
        self.pred_len = pred_len
        self.kernelSize = kernelSize
        self.dilation = dilation
        self.dropout = dropout
        self.isDepthWise = isDepthWise
        """
        conv1d_module = [
        nn.ReplicationPad1d((pad_l, pad_r)),
        nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                  kernel_size=self.kernel_size, dilation=self.dilation,
                  stride=1, groups=self.groups),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.Dropout(self.dropout),
        nn.Conv1d(int(in_planes * size_hidden), in_planes,
                    kernel_size=3, stride=1, groups=self.groups
                    ),
        nn.Tanh()
        ]
        nn.seqeuential(*conv1d_module)
        # TODO: last layer : mlp to aggregate all timestep features
        """
        layers = OrderedDict()
        
        pad = dilation * (kernelSize - 1) // 2
        # stride is 1 for same input length and output length

        # hidden layers
        for idx, (in_channels, out_channels) in enumerate(features):
            layer_name = f'conv1d_{idx + 1}'
            layers[layer_name] = nn.Conv1d(
                in_channels,
                in_channels * out_channels if isDepthWise else out_channels,
                padding=pad,
                padding_mode='replicate',
                kernel_size=kernelSize,
                dilation=dilation,
                groups=in_channels if isDepthWise else 1
                )
            
            act1_name = f'LeakyReLU_{idx + 1}'
            layers[act1_name] = nn.LeakyReLU(negative_slope=0.01, inplace=True)
            
            drop_name = f'Dropout_{idx + 1}'
            layers[drop_name] = nn.Dropout(dropout)

            layer_name2 = f'conv1d-2_{idx + 1}'
            layers[layer_name2] = nn.Conv1d(
                in_channels * out_channels if isDepthWise else out_channels,
                out_channels,
                padding=pad,
                padding_mode='replicate',
                kernel_size=3, 
                stride=1, 
                groups=1
                )
            
            act2_name = f'Tanh_{idx + 1}'
            layers[act2_name] = nn.Tanh()
            # local windowing에 의한 차이를 얘기할 수 있으려면 length는 유지되야함.
            # causal convolution? No.내 자료는 이미 과거/미래 분리가 되어있으므로 필요없음.
            
        self.layers = nn.Sequential(layers)
        self.flatten = nn.Flatten()
        
        last_dim = features[-1][1]  # features : (in_ch, out_ch, ker_size)
        self.output = nn.Linear(last_dim * input_len, pred_len)
        self.relu = nn.ReLU()


    def forward(self, x):
        # (N, C, T) for pytorch

        # RuntimeError: Given groups=11, weight of size [1408, 1, 3], 
        # expected input[32, 24, 13] to have 11 channels, but got 24 channels instead
        hidden = self.layers(x)
        flatten = self.flatten(hidden)
        out = self.output(flatten)
        out = self.relu(out)

        return out