import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('ignore')
from data_process import (NIA_data_loader_csvOnly_YearSplit,
                          NIA_data_loader_jsonRead)
from experiments.exp_basic import Exp_Basic
from metrics.NIA_metrics import metric, score_in_1h

from models.DNN import DNN
from models.SCINet_decompose import SCINet_decomp
from utils.tools import (EarlyStopping, adjust_learning_rate, load_model,
                         save_model)


class Experiment_DL(Exp_Basic):
    def __init__(self, args):
        super(Experiment_DL, self).__init__(args)
        self.print_per_iter = 100;

        # train / val / test dataset and dataloader setting
        if args.nia_csv_base:
            module = NIA_data_loader_csvOnly_YearSplit.Dataset_NIA
        else:
            module = NIA_data_loader_jsonRead.Dataset_NIA
            
        train_set, val_set, test_set = module(
                                          root_path = args.root_path,
                                          NIA_work = args.NIA_work,
                                          data = args.data,
                                          port = args.port,
                                          data_path = args.data_path,
                                          size = [args.seq_len, args.pred_len],
                                          args = args
                                       )
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

        #FIXME: for debug, check time range overlap or shape
        #FIXME: analyze statistics of train/val/test set

        self.train_loader = DataLoader(
                                train_set,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                drop_last=True
                            )
        self.val_loader =   DataLoader(
                                val_set,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                drop_last=True
                            )
        self.test_loader =  DataLoader(
                                test_set,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                drop_last=False
                            )


    def _build_model(self):
        assert self.args.model_name not in ['SARIMAX', 'RF', 'XGB']

        if self.args.model_name == 'DNN':
            model = DNN(
                        features=[
                                  (self.args.seq_len * self.args.in_dim, 512), 
                                  (512, 1024), 
                                  (1024, 2048), 
                                  (2048, 1024),
                                  (1024, 128),
                                 ],
                        pred_len=self.args.pred_len
                    )
        elif self.args.model_name == 'CNN':
            ...
        elif self.args.model_name == 'RNN':
            ...
        elif self.args.model_name == 'SCINet':  # and self.args.decompose:
            model = SCINet_decomp(
                        output_len=self.args.pred_len,
                        input_len=self.args.seq_len,
                        input_dim= self.args.in_dim,
                        hid_size=self.args.hidden_size,
                        num_stacks=self.args.stacks,
                        num_levels=self.args.levels,
                        concat_len=self.args.concat_len,
                        groups=self.args.groups,
                        kernel=self.args.kernel,
                        dropout=self.args.dropout,
                        single_step_output_One=self.args.single_step_output_One,
                        positionalE=self.args.positionalEcoding,
                        modified=True,
                        RIN=self.args.RIN
                    )
        elif self.args.model_name == 'Transformer':
            ...

        return model.double()  # double() 이 뭐하는 함수고?


    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return model_optim
    

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
        path = os.path.join(self.args.checkpoints, setting)
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)
        
        writer = SummaryWriter(f'event/run_{self.args.data}/{self.args.model_name}')

        time_now = time.time()
        
        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion(self.args.loss)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        if self.args.resume:
            self.model, lr, epoch_start = load_model(
                                                    self.model, path, 
                                                     model_name=self.args.data, 
                                                     horizon=self.args.horizon
                                          )
        else:
            epoch_start = 0

        for epoch in range(epoch_start, self.args.train_epochs):
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y) in enumerate(self.train_loader):
                model_optim.zero_grad()
                
                pred, pred_scale, mid, mid_scale, true, true_scale = \
                           self._process_one_batch(self.train_data, batch_x, batch_y)
                if self.args.model_name == 'SCINet':  
                    loss = criterion(pred, true) + criterion(mid, true)
                else:
                    loss = criterion(pred, true)
                                     
                train_loss.append(loss.item())
                
                if (i + 1) % self.print_per_iter == 0:
                    speed = (time.time() - time_now) / self.print_per_iter
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} | speed: {3:.4f}s/iter".format(
                                    i + 1, epoch + 1, loss.item(), speed))
                    # left_time = speed*((self.args.train_epochs - epoch) * train_steps - i)
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    time_now = time.time()
                
                if self.args.use_amp:
                    print('use amp')    
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            print('--------start to validate-----------')
            valid_loss = self.valid(self.valid_data, self.valid_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} valid Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, valid_loss))

            writer.add_scalar('train_loss', train_loss, global_step=epoch)
            writer.add_scalar('valid_loss', valid_loss, global_step=epoch)

            early_stopping(valid_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            lr = adjust_learning_rate(model_optim, epoch + 1, self.args)
        
        if self.args.evaluate:
            print('--------start to test-----------')
            test_loss = self.valid(self.test_data, self.test_loader, criterion)
            print("Test Loss: {:.7f}".format(test_loss))

        save_model(epoch, lr, self.model, path, model_name=self.args.data, 
                                                horizon=self.args.pred_len)
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model


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

        for i, (batch_x, batch_y) in enumerate(valid_loader):
            pred, pred_scale, mid, mid_scale, true, true_scale = \
                           self._process_one_batch(valid_data, batch_x, batch_y)

            if self.args.model_name == 'SCINet':
                loss = criterion(pred.detach().cpu(), 
                                 true.detach().cpu()) + \
                                    criterion(mid.detach().cpu(), 
                                              true.detach().cpu()
                                )

            else:
                loss = criterion(pred.detach().cpu(), true.detach().cpu())

            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            pred_scales.append(pred_scale.detach().cpu().numpy())
            true_scales.append(true_scale.detach().cpu().numpy())
            if self.args.model_name == 'SCINet':
                mids.append(mid.detach().cpu().numpy())
                mid_scales.append(mid_scale.detach().cpu().numpy())
            else:
                mids.append(0)
                mid_scales.append(0)
            
            total_loss.append(loss)
        total_loss = np.average(total_loss)

        true_scales = np.array(true_scales)
        pred_scales = np.array(pred_scales)

        true_scales = true_scales.reshape(-1, 
                                          true_scales.shape[-2], 
                                          true_scales.shape[-1])
        pred_scales = pred_scales.reshape(-1, 
                                          pred_scales.shape[-2], 
                                          pred_scales.shape[-1])

        print('==== Final ====')
        acc, f1, acc_1h, f1_1h, true, pred, true_1h, pred_1h = \
                                      score_in_1h(pred_scales, true_scales, self.args)
        print(f'Accuracy, F1 in 1h: {acc_1h :.3f}, {f1_1h :.3f}')
        print(f'Accuracy, F1: {acc :.3f}, {f1 :.3f} \n\n')

        return total_loss
    

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
        
        if evaluate:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        for i, (batch_x,batch_y) in enumerate(self.test_loader):
            pred, pred_scale, mid, mid_scale, true, true_scale = \
                            self._process_one_batch(self.test_data, batch_x, batch_y)

            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            pred_scales.append(pred_scale.detach().cpu().numpy())
            true_scales.append(true_scale.detach().cpu().numpy())

        pred_scales = np.array(pred_scales)
        true_scales = np.array(true_scales)

        true_scales = true_scales.reshape(-1, 
                                          true_scales.shape[-2], 
                                          true_scales.shape[-1])
        pred_scales = pred_scales.reshape(-1, 
                                          pred_scales.shape[-2], 
                                          pred_scales.shape[-1])
        
        print('==== Final ====')
        acc, f1, acc_1h, f1_1h, true, pred, true_1h, pred_1h = \
                                score_in_1h(pred_scales, true_scales, self.args)
        print(f'Accuracy, F1 in 1h: {acc_1h :.3f}, {f1_1h :.3f}\n\n')        

        return acc, f1, acc_1h, f1_1h
        

    def _process_one_batch(self, 
                           dataset_object,
                           batch_x, 
                           batch_y
                          ) -> tuple:
        """
        one batch process for train, val, test
        """
        batch_x = batch_x.double().cuda()
        batch_y = batch_y.double()

        f_dim = 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].cuda()

        if self.args.model_name == 'SCINet':
            outputs, mid = self.model(batch_x)

            assert self.args.stacks == 2, "SCINet stack size is supposed to be larger than 1"

            # Except one hot encoding
            outputs = outputs[..., :-5]
            mid = mid[..., :-5]
            batch_y = batch_y[..., :-5]

            outputs_scaled = dataset_object.inverse_transform(outputs)
            mid_scaled = dataset_object.inverse_transform(mid)
            batch_y_scaled = dataset_object.inverse_transform(batch_y)

            return outputs[:,:,-1], outputs_scaled[:,:,-1], mid[:,:,-1], \
                   mid_scaled[:,:,-1], batch_y[:,:,-1], batch_y_scaled[:,:,-1]
            
        elif self.args.model_name == 'DNN':
            batch_x = batch_x.reshape((-1, batch_x.shape[1] * batch_x.shape[2]))
            outputs = self.model(batch_x)
            batch_y = batch_y[..., 10]

            outputs_scaled = dataset_object.inverse_transform(outputs, is_dnn=True)
            batch_y_scaled = dataset_object.inverse_transform(batch_y, is_dnn=True)

            return outputs, outputs_scaled, 0, 0, batch_y, batch_y_scaled
        else:
            print()
