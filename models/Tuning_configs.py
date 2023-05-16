import numpy as np


def model_tuning_dict(args: object) -> dict:

    if args.model_name == 'MLPvanilla':
        tuning_dict = {
            'n_hidden_layer': np.arange(5, 12 + 1, step=3),  #FIXME: 3부터
            'dropRate': np.arange(0.6, 1.01, step=0.2),
            'out_features': 2 ** np.arange(7, 9, step=1),
            'expand': np.arange(1.5, 2.1, step=0.5),
        }

    elif args.model_name == 'Simple1DCNN':
        tuning_dict = {
            'max_depth': np.arange(3, 31, step=3),
            'max_leaves': np.arange(1, 31, step=3),
            'max_depth': np.arange(3, 30, step=2),
        }

    elif args.model_name == '1DCNN':
        tuning_dict = {
            'eta': np.arange(0.05, 0.3, step=0.05),
            'gamma': np.arange(0, 0.02, step=0.01),
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

    elif args.model_name == 'SimpleLinear':
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
