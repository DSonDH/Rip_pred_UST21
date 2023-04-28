import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
warnings.filterwarnings('ignore')
from data_process import NIA_data_loader_csvOnly_YearSplit
                          
from experiments.exp_basic import Exp_Basic

from typing import Any

class Experiment_ML(Exp_Basic):
    """ Overall experiment pipelines are implemented in OOP style
    __init__()
        : prepare data
    _build_model()
        : generate model objects
    _select_optimizer()
        : call optimization function
    _select_criterion()
        : call loss function
    train()
        : do train process
    valid()
    test()
        FIXME: valid, test 중복되는거 같은데 중복 없애기
        FIXME: HPO: grid search로
    _process_one_batch()
        : get ground truth, prediction result of one batch
        both with scaled and without scaled
    """
    def __init__(self, args):
        super(Experiment_ML, self).__init__(args)
        self.print_per_iter = 100;

        DatasetClass = NIA_data_loader_csvOnly_YearSplit.Dataset_NIA_class
            
        self.dataset = DatasetClass(
                           root_path = args.root_path,
                           NIA_work = args.NIA_work,
                           data = args.data,
                           port = args.port,
                           data_path = args.data_path,
                           size = [args.input_len, args.pred_len],
                           args = args
                       )

        self.train_loader = DataLoader(
                                self.dataset.train_set,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                drop_last=True
                            )
        self.val_loader =   DataLoader(
                                self.dataset.val_set,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                drop_last=True
                            )
        self.test_loader =  DataLoader(
                                self.dataset.test_set,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                drop_last=False
                            )


    def __call__(self, args: Any, **kwds: Any) -> Any:
        """main funciton of this Experimental_DL class object
        Args:
            args
        Return:
            (y_test_label, pred_test)
        """
        #TODO: complete this call method
        
        if args.do_train:
            self.train(args)
        y_test_label, pred_test = self.test(args)

        return y_test_label, pred_test


    def _build_model(self):
        assert self.args.model_name in ['RF', 'XGB']

        if self.args.model_name == 'DNN':
            model = 1
        elif self.args.model_name == 'CNN':
            ...
        return 


    def _select_criterion(self, losstype):
        if losstype == "mse":
            criterion = nn.MSELoss()
        elif losstype == "mae":
            criterion = nn.L1Loss()
        elif losstype == "BCE":
            criterion = nn.BCELoss()
        else:
            criterion = nn.L1Loss()
        return criterion


    def train(self, setting):
        """ do train
        """
        ...

    def valid(self, valid_data, valid_loader, criterion):
        """
        do validation
        """
        self.model.eval()
        total_loss = []
        preds = []
        trues = []
        mids = []
        pred_scales = []
        true_scales = []
        mid_scales = []


    

    def test(self, setting, evaluate=False):
        """test

        """
        self.model.eval()
        
        preds = []
        trues = []
        mids = []
        pred_scales = []
        true_scales = []
        mid_scales = []
