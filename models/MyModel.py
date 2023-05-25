from collections import OrderedDict
from torch import nn


class CNN1D(nn.Module):
    """ simple 1dnn
    It only treats depth, hidden usit size, dropout rate.
    
    simpleDNN과의 차이점은 
    convolution 연산에 의해 local feature 추출이 용이한 것
    """
    def __init__(self, 
                 features: list, 
                 pred_len: int, 
                 dropout:float = 0.5,
                 positionalE:bool = False, 
                ):
        
        super(CNN1D, self).__init__()

        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, 
                                stride=1, padding=0, dilation=1, 
                                groups=1, bias=True, padding_mode='zeros', 
                                device=None, dtype=None)

        #TODO: conv1d, pooling1d, dropout or layerNorm1d, skip-connection, 
        # sparsity?,  1x1 convolution, etc. 기타 실험하고픈거 있으면 직접 모델 구축해서 하기
        layers = OrderedDict()
        for idx, params in enumerate(features):
            layer_name = f'conv1d_{idx + 1}'
            layers[layer_name] = nn.Conv1d(*params)
            
            activation_name = f'LeakyReLU_{idx + 1}'
            layers[activation_name] = nn.LeakyReLU()
            
            dropout_name = f'dropout_{idx + 1}'
            layers[dropout_name] = nn.Dropout(dropout)
            
        self.layers = nn.Sequential(layers)
        last_dim = features[-1][-1]
        
        # TODO: last mlp for prediction
        # or fully convolutional layer로 ?
        self.output = nn.Linear(last_dim, pred_len)

    #TODO: weight initialization? dnn도 weight 초기화 바꿀지 결정해야함.


    def forward(self, x):
        # PE = self.get_position_encoding(x)
        # x = x.reshape((-1, x.shape[1] * x.shape[2]))
        hidden = self.layers(x)
        out = self.output(hidden)
        out = nn.ReLU(out)

        return out