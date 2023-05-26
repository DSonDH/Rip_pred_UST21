import numpy as np


def model_tuning_dict(args: object) -> dict:
    if args.model_name == 'MLPvanilla':
        tuning_dict = {
            'n_layers': np.arange(3, 7 + 1, step=1),
            'n_hidden_units': 2 ** np.arange(7, 12 + 1, step=1),
            'dropRate': np.arange(0.6, 1.01, step=0.2),
        }

    elif args.model_name == 'Simple1DCNN':
        # tmp_len = (tmp_len + 2 * padding - dilation * (kernel_size - 1) \
        # + stride) // stride
        tuning_dict = {
            'n_layers': np.arange(3, 11 + 1, step=2),
            'out_channel': 2 ** np.arange(7, 12, step=1),
            'isDepthWise': [True, False],
            'dropRate': np.arange(0.6, 1.01, step=0.2),
            'kernelSize': np.arange(3, 9 + 1, step=2),  # 5분간격
            'stride': [1],
            'dilation': np.arange(1, 3 + 1, step=1),
        }
        
    elif args.model_name == 'LSTM':
        tuning_dict = {
            'eta': np.arange(0.05, 0.3, step=0.05),
            'gamma': np.arange(0, 0.02, step=0.01),
        }

    elif args.model_name == 'Transformer':
        tuning_dict = {
            'eta': np.arange(0.05, 0.3, step=0.05),
            'gamma': np.arange(0, 0.02, step=0.01),
        }

    elif args.model_name == 'LTSF-Linear':
        tuning_dict = {
            'eta': np.arange(0.05, 0.3, step=0.05),
            'gamma': np.arange(0, 0.02, step=0.01),
        }

    elif args.model_name == 'LightTS':
        tuning_dict = {
            'eta': np.arange(0.05, 0.3, step=0.05),
            'gamma': np.arange(0, 0.02, step=0.01),
        }

    elif args.model_name == 'SCINet':
        tuning_dict = {
            'lr': np.arange(0.05, 0.3, step=0.05),
            'hidden-size': 32,
            'hidden-size': 1,  # hidden channel of module
            'INN': 1, # use INN or basic strategy
            'kernel': 5, # kernel size, 3, 5, 7
            'dilation': 1,
            'window_size': 3,  # input size
            'dropout': 0.5,
            'positionalEcoding': False,
            'groups': 2,
            'levels': 2,
            'stacks': 2, # 1 stack or 2 stacks
            'num_decoder_layer': 1,
            'RIN': False,
            'decompose': True,
        }

    elif args.model_name == 'Informer':
        tuning_dict={
            'eta': np.arange(0.05, 0.3, step=0.05),
            'gamma': np.arange(0, 0.02, step=0.01),
        }

    else:
        raise Exception

    return tuning_dict
