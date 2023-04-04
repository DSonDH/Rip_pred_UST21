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
from data_process import (NIA_KHOA_data_loader_csvOnly,
                          NIA_KHOA_data_loader_jsonRead)
from experiments.exp_basic import Exp_Basic
from metrics.NIA_KHOA_metrics import metric, score_in_1h

from models.DNN import DNN
from models.SCINet import SCINet
from models.SCINet_decompose import SCINet_decomp
from utils.tools import (EarlyStopping, adjust_learning_rate, load_model,
                         save_model)


class Experiment(Exp_Basic):
    def __init__(self, args):
        super(Experiment, self).__init__(args)
    

    def _build_model(self):
        if self.args.model_name == 'DNN':
            model = DNN(
                        features=[
                                  (self.args.in_dim, 100), 
                                  (100, 1000), 
                                  (1000, 1000), 
                                  (1000, 100)
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


        return model.double()


    def _get_data(self, flag):
        args = self.args
        if args.nia_csv_base:
            module = NIA_KHOA_data_loader_csvOnly.Dataset_NIA_KHOA
        else:
            module = NIA_KHOA_data_loader_jsonRead.Dataset_NIA_KHOA
            
        data_set = module(
            root_path = args.root_path,
            NIA_work = args.NIA_work,
            data = args.data,
            port = args.port,
            data_path = args.data_path,
            flag = flag,
            size = [args.seq_len, args.pred_len],
            args = args
        )

        shuffle_flag = True; drop_last = True; batch_size = args.batch_size
        # if flag == 'test':
        #     shuffle_flag = False; drop_last = True; batch_size = args.batch_size
        # else:
        #     shuffle_flag = True; drop_last = True; batch_size = args.batch_size

        print(flag, len(data_set))
        data_loader = DataLoader(
                          data_set,
                          batch_size=batch_size,
                          shuffle=shuffle_flag,
                          num_workers=args.num_workers,
                          drop_last=drop_last
                      )
        return data_set, data_loader


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


    def valid(self, valid_data, valid_loader, criterion):
        self.model.eval()
        total_loss = []
        preds = []
        trues = []
        mids = []
        pred_scales = []
        true_scales = []
        mid_scales = []

        for i, (batch_x, batch_y) in enumerate(valid_loader):
            pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch(
                valid_data, batch_x, batch_y)

            if self.args.stacks == 1:
                loss = criterion(pred.detach().cpu(), true.detach().cpu())

                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                pred_scales.append(pred_scale.detach().cpu().numpy())
                true_scales.append(true_scale.detach().cpu().numpy())

            elif self.args.stacks == 2:
                loss = criterion(pred.detach().cpu(), true.detach().cpu()) + criterion(mid.detach().cpu(), true.detach().cpu())

                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                mids.append(mid.detach().cpu().numpy())
                pred_scales.append(pred_scale.detach().cpu().numpy())
                mid_scales.append(mid_scale.detach().cpu().numpy())
                true_scales.append(true_scale.detach().cpu().numpy())

            else:
                print('Error!')

            total_loss.append(loss)
        total_loss = np.average(total_loss)

        preds = np.array(preds)
        trues = np.array(trues)
        mids = np.array(mids)
        mid_scales = np.array(mid_scales)
        true_scales = np.array(true_scales)
        pred_scales = np.array(pred_scales)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        mids = mids.reshape(-1, mids.shape[-2], mids.shape[-1])
        mid_scales = mid_scales.reshape(-1, mid_scales.shape[-2], mid_scales.shape[-1])
        true_scales = true_scales.reshape(-1, true_scales.shape[-2], true_scales.shape[-1])
        pred_scales = pred_scales.reshape(-1, pred_scales.shape[-2], pred_scales.shape[-1])
        # print('test shape:', preds.shape, mids.shape, trues.shape)

        print('==== Final ====')
        acc_mid, f1_mid, acc_1h_mid, f1_1h_mid, true, pred, true_1h, pred_1h = \
                                     score_in_1h(mid_scales, true_scales, self.args)
        print(f'Accuracy, F1 in 1h: {acc_1h_mid :.3f}, {f1_1h_mid :.3f}')
        print(f'Accuracy, F1: {acc_mid :.3f}, {f1_mid :.3f}\n')

        print('==== Final ====')
        acc, f1, acc_1h, f1_1h, true, pred, true_1h, pred_1h = \
                                      score_in_1h(pred_scales, true_scales, self.args)
        print(f'Accuracy, F1 in 1h: {acc_1h :.3f}, {f1_1h :.3f}')
        print(f'Accuracy, F1: {acc :.3f}, {f1 :.3f} \n\n')

        return total_loss


    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        valid_data, valid_loader = self._get_data(flag = 'val')

        if self.args.evaluate:
            test_data, test_loader = self._get_data(flag = 'test')
        
        path = os.path.join(self.args.checkpoints, setting)
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)
        
        writer = SummaryWriter(f'event/run_{self.args.data}/{self.args.model_name}')
        

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion(self.args.loss)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        if self.args.resume:
            self.model, lr, epoch_start = load_model(self.model, path, model_name=self.args.data, horizon=self.args.horizon)
        else:
            epoch_start = 0

        for epoch in range(epoch_start, self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            # for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch(
                    train_data, batch_x, batch_y)

                if self.args.stacks == 1:
                    loss = criterion(pred, true)
                elif self.args.stacks == 2:
                    loss = criterion(pred, true) + criterion(mid, true)
                else:
                    print('Error!')

                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    speed = (time.time()-time_now)/iter_count
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} | speed: {3:.4f}s/iter".format(i + 1, epoch + 1, loss.item(), speed))
                    # left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    print('use amp')    
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            print('--------start to validate-----------')
            valid_loss = self.valid(valid_data, valid_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} valid Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, valid_loss))

            writer.add_scalar('train_loss', train_loss, global_step=epoch)
            writer.add_scalar('valid_loss', valid_loss, global_step=epoch)

            early_stopping(valid_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            lr = adjust_learning_rate(model_optim, epoch+1, self.args)
        
        if self.args.evaluate:
            print('--------start to test-----------')
            test_loss = self.valid(test_data, test_loader, criterion)
            print("Test Loss: {:.7f}".format(test_loss))

        save_model(epoch, lr, self.model, path, model_name=self.args.data, horizon=self.args.pred_len)
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model


    def test(self, setting, evaluate=False):
        test_data, test_loader = self._get_data(flag='test')
        
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

        for i, (batch_x,batch_y) in enumerate(test_loader):
        # for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch(
                test_data, batch_x, batch_y)

            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            mids.append(mid.detach().cpu().numpy())
            pred_scales.append(pred_scale.detach().cpu().numpy())
            mid_scales.append(mid_scale.detach().cpu().numpy())
            true_scales.append(true_scale.detach().cpu().numpy())

        pred_scales = np.array(pred_scales)
        true_scales = np.array(true_scales)
        mid_scales = np.array(mid_scales)

        true_scales = true_scales.reshape(-1, true_scales.shape[-2], true_scales.shape[-1])
        pred_scales = pred_scales.reshape(-1, pred_scales.shape[-2], pred_scales.shape[-1])
        mid_scales = mid_scales.reshape(-1, mid_scales.shape[-2], mid_scales.shape[-1])
        
        #TODO: csv output 산출 코드 작성
        import pandas as pd
        df = pd.DataFrame()
        site_names = []
        for site in ['DC', 'HD', 'JM', 'NS', 'SJ']:
            df_te = pd.read_csv(f'datasets/NIA/meta_csv/{site}_test.csv',
                                        encoding='euc-kr')
            site_names.extend([site] * len(df_te))
            df = df.append(df_te)
        times = df['ob time']
        site_times = [site +' '+ times.iloc[i] 
                      for i, site in enumerate(site_names)]
        # csv다시 쌓은거랑, testset rescale한거랑 같음
        # df[df.columns[1:]]
        # test = test_data.inverse_transform(test_data.X[:, 0, :11])
        # sum(df[df.columns[1:]].values != test)

        print('---- Mid ----')
        acc_mid, f1_mid, acc_1h_mid, f1_1h_mid, true, pred, true_1h, pred_1h = \
                             score_in_1h(mid_scales, true_scales, self.args)
        # print(f'Accuracy, F1 : {acc_mid :.3f}, {f1_mid :.3f}')
        print(f'Accuracy, F1 in 1h: {acc_1h_mid :.3f}, {f1_1h_mid :.3f}\n')
        
        print('==== Final ====')
        acc, f1, acc_1h, f1_1h, true, pred, true_1h, pred_1h = \
                            score_in_1h(pred_scales, true_scales, self.args)
        # print(f'Accuracy, F1: {acc :.3f}, {f1 :.3f}')
        print(f'Accuracy, F1 in 1h: {acc_1h :.3f}, {f1_1h :.3f}\n\n')

        # save result to csv
        df1 = pd.DataFrame(columns = ['ID', 'TP', 'FP', 'FN', 'PAG', 'POD', 'F1'])
        df1.ID = site_times
        df1.TP = np.cumsum(
                 [1 if (true_1h[i] == 1) and (pred_1h[i] == 1) else 0 for i in range(len(df))]
                 )
        df1.FP = np.cumsum(
                 [1 if (true_1h[i] == 0) and (pred_1h[i] == 1) else 0 for i in range(len(df))]
                 )
        df1.FN = np.cumsum(
                 [1 if (true_1h[i] == 1) and (pred_1h[i] == 0) else 0 for i in range(len(df))]
                 )
        df1.PAG = [df1.TP[i] / (df1.TP[i] + df1.FP[i]) if (df1.TP[i] + df1.FP[i]) != 0 else '-'
                   for i in range(len(df))
                  ]
        df1.POD = [df1.TP[i] / (df1.TP[i] + df1.FN[i]) if (df1.TP[i] + df1.FN[i]) != 0 else '-'
                   for i in range(len(df))
                  ]
        df1.F1 =  [2 * df1.PAG[i] *df1.POD[i] / (df1.PAG[i] + df1.POD[i]) 
                   if (df1.PAG[i] != '-') and (df1.POD[i] != '-') else '-'
                   for i in range(len(df))
                  ]
        
        df2 = pd.DataFrame(columns = ['ID', 'GT', 'Pred'])
        df2.ID = site_times
        df2.GT = true_1h
        df2.Pred = pred_1h

        df1.to_csv(f'./NIA_ripPred_test_result_scores.csv')
        df2.to_csv(f'./NIA_ripPred_test_result_labels.csv')
        return acc, f1, acc_1h, f1_1h


    def _process_one_batch(self, dataset_object, batch_x, batch_y):
        batch_x = batch_x.double().cuda()
        batch_y = batch_y.double()

        f_dim = 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].cuda()

        if self.args.model_name == 'SCINet':
            outputs, mid = self.model(batch_x)
            
            assert(self.args.stacks == 2, 
                   "SCINet stack size is supposed to be larger than 1")

            # Except one hot encoding
            outputs = outputs[..., :-5]
            mid = mid[..., :-5]
            batch_y = batch_y[..., :-5]

            outputs_scaled = dataset_object.inverse_transform(outputs)
            mid_scaled = dataset_object.inverse_transform(mid)
            batch_y_scaled = dataset_object.inverse_transform(batch_y)

            return outputs[:,:,-1], outputs_scaled[:,:,-1], mid[:,:,-1], mid_scaled[:,:,-1], batch_y[:,:,-1], batch_y_scaled[:,:,-1]
            
        else:
            outputs = self.model(batch_x)

            # Except one hot encoding
            outputs = outputs[..., :-5]
            batch_y = batch_y[..., :-5]

            outputs_scaled = dataset_object.inverse_transform(outputs)
            batch_y_scaled = dataset_object.inverse_transform(batch_y)

            return outputs[:,:,-1], outputs_scaled[:,:,-1], 0, 0, batch_y[:,:,-1], batch_y_scaled[:,:,-1]
            