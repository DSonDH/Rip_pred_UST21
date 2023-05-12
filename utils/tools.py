from typing import List, Set, Union
import os
import numpy as np
import torch
import pandas as pd


def save_model(epoch:int, lr:float, model:object, save_path:str) -> None:
    torch.save({'epoch': epoch,
                'lr': lr,
                'model': model.state_dict()},
               save_path
               )
    print('saved model in ', save_path)


def load_model(model, model_dir, model_name, pred_len):
    if not model_dir:
        return
    file_name = os.path.join(model_dir, model_name+str(pred_len)+'.pt')

    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        checkpoint = torch.load(f, map_location=lambda storage, loc: storage)
        print('This model was trained for {} epochs'.format(
            checkpoint['epoch']))
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        lr = checkpoint['lr']
        print('loaded the model...', file_name,
              'now lr:', lr, 'now epoch:', epoch)
    return model, lr, epoch


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 1:
        lr_adjust = {epoch: args.lr * (0.95 ** (epoch // 1))}

    elif args.lradj == 2:
        lr_adjust = {
            0: 0.0001, 5: 0.0005, 10: 0.001, 20: 0.0001, 30: 0.00005, 40: 0.00001, 70: 0.000001
        }

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
    else:
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
    return lr


class EarlyStopping:
    def __init__(self, patience:int = 7, delta: float = 0., 
                 checkLoss:bool = True, verbose: bool = True):
        self.patience = patience
        self.delta = delta
        self.checkLoss = checkLoss  # is it loss to watch? acc or loss
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.best_score = np.inf if checkLoss else 0.
        

    def __call__(self, score:float) -> None:
        """inspect early stopping criteria and behave"""

        if self.checkLoss:
            is_improved = (score - self.best_score) < -self.delta
        else:
            is_improved = (score - self.best_score) > self.delta
        
        if is_improved:  # good epoch
            if self.verbose:
                print(f'Validation improved!! '\
                      f'({self.best_score:.6f} --> {score:.6f}).'\
                      f' Saving model ...')
            self.best_score = score
            self.counter = 0

        else:  # bad epoch
            self.counter += 1
            if self.verbose:
                print(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
                
            if self.counter >= self.patience:
                self.early_stop = True
        

# class dotdict(dict):
#     """dot.notation access to dictionary attributes"""
#     __getattr__ = dict.get
#     __setattr__ = dict.__setitem__
#     __delattr__ = dict.__delitem__


class StandardScaler():
    """
    2d (N, T * C)로 normalization 하지 않고
    3d (N, T, C)로 모든 T에 동일하게 scaling하여
    1d (C) mean, std를 얻도록 한다.

    label (index 10)도 함께 scaling하므로, 나중에 꼭 inverse scaling 해야함.
    """

    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data: np.array):
        assert data.ndim == 3, 'array should have 3d (B, T, C) channels'
        self.mean = data.mean(axis=(0, 1))
        self.std = data.std(axis=(0, 1))

        assert self.mean.ndim == 1
        assert self.std.ndim == 1

    def transform(self, data: np.array):
        assert data.ndim == 3, 'array should have 3d (B, T, C) channels'
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) \
            if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) \
            if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data: np.array):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) \
            if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) \
            if torch.is_tensor(data) else self.std

        if data.ndim == 3:
            # N x T x C
            assert data.shape[2] > 1, 'number of features are expected to be full'
            return (data * std) + mean
        else:
            # N x T
            return (data * std[-1]) + mean[-1]


def print_performance(model_name: str, metrics: dict) -> None:
    print('*'*41)
    print(f'Final test metrics of {model_name}:')
    for key in metrics:
        print(f"{key}: {metrics[key]}")
    print('*'*41)


def record_studyname_metrics(df: pd.DataFrame, study_name: str, metrics: dict
                             ) -> pd.DataFrame:
    idx = len(df)
    df.loc[idx, 'study_name'] = study_name
    for item in metrics:
        df.loc[idx, item] = metrics[item]

    return df
