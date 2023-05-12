from typing import Any
from utils.tools import (EarlyStopping, adjust_learning_rate, load_model,
                         save_model)
from models.SCINet_decompose import SCINet_decomp
from models.DNN import DNN
from data_process import NIA_data_loader_csvOnly_YearSplit
import itertools

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


class Experiment_DL():
    """ Overall experiment pipelines are implemented in OOP style
    __init__()
        : prepare data
    _build_model()
        : generate model objects
    _select_optimizer()
        : call optimization function
    _select_criterion()
        : call loss function
    _process_one_batch()
        : get ground truth, prediction result of one batch
        only scaled data
    train()
        : do train process
    valid()
    test()

    !!!! train / validation loss 시계열 예측 task에서 loss는 곧 acc다. 
    clasf같은 경우는 loss나 f1이나 다를 수 있지만... 
    따라서 acc, loss둘 다 모니터링 하는건 낭비임 !!!!
    """

    def __init__(self, args, hp):
        self.print_per_iter = 100
        self.args = args
        
        self.model = self._build_model(hp).cuda()

        self.dataset_train = NIA_data_loader_csvOnly_YearSplit.Dataset_NIA_class(
            args=args,
            flag='train',
            is_2d=False
        )
        self.dataset_val = NIA_data_loader_csvOnly_YearSplit.Dataset_NIA_class(
            args=args,
            flag='val',
            is_2d=False
        )
        self.dataset_test = NIA_data_loader_csvOnly_YearSplit.Dataset_NIA_class(
            args=args,
            flag='test',
            is_2d=False
        )

        self.train_loader = DataLoader(
            self.dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True
        )
        self.val_loader = DataLoader(
            self.dataset_val,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True
        )
        self.test_loader = DataLoader(
            self.dataset_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False
        )


    def _build_model(self, hp: dict):
        # FIXME: parameter setting using hp variables
        if self.args.model_name == 'MLPvanilla':
            model = DNN(
                features=[
                    (self.args.input_len * self.args.input_dim, 512),
                    (512, 1024),
                    (1024, 2048),
                    (2048, 1024),
                    (1024, 128),
                ],
                pred_len=self.args.pred_len
            )
        elif self.args.model_name == 'Simple1DCNN':
            ...
            # TODO:FIXME: search basic 1dcnn model and test it

        elif self.args.model_name == 'LSTM':
            # FIXME:TODO: search basic 1dcnn model and test it
            ...
        elif self.args.model_name == 'Transformer':
            # TODO:FIXME: search basic 1dcnn model and test it
            ...
        elif self.args.model_name == 'SimpleLinear':
            # FIXME:TODO: search basic 1dcnn model and test it
            ...
        elif self.args.model_name == 'LightTS':
            # TODO:FIXME: search basic 1dcnn model and test it
            ...
        elif self.args.model_name == 'SCINet':  # and self.args.decompose:
            model = SCINet_decomp(
                output_len=self.args.pred_len,
                input_len=self.args.input_len,
                input_dim=self.args.input_dim,
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
        return model.double()  # convert model's parameter dtype to double


    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(),
                                 lr=self.args.lr
                                 )
        # FIXME: AdamW ?? or any other options as HPO
        return model_optim


    def _select_loss(self, losstype):
        if losstype == "mse":
            criterion = nn.MSELoss()
        elif losstype == "mae":
            criterion = nn.L1Loss()
        else:
            raise Exception
        return criterion


    def _process_one_batch(self, scaler, batch_x, batch_y) -> tuple:
        """one batch process for train, val, test
        코드 확인 포인트: scaler 적용 순서가 맞는지, X, y, pred shape이 맞는지
        """
        batch_x = batch_x.double().cuda()
        batch_y = batch_y[..., -1].double().cuda()

        batch_x = batch_x.reshape((-1, batch_x.shape[1] * batch_x.shape[2]))

        if self.args.model_name == 'SCINet':
            pred, pred_mid = self.model(batch_x)
            pred_scaled = scaler.inverse_transform(pred)
            pred_mid_scaled = scaler.inverse_transform(pred_mid)
        else:
            pred = self.model(batch_x)
            pred_scaled = scaler.inverse_transform(pred)
            pred_mid_scaled = None

        batch_y_scaled = scaler.inverse_transform(batch_y)

        return batch_y_scaled, pred_scaled, pred_mid_scaled


    def get_validation_loss(self, valid_loader, loss_fn) -> float:
        self.model.eval()

        total_loss = []
        for batch_x, batch_y in valid_loader:

            true_scale, pred_scale, mid_scale = \
                self._process_one_batch(self.dataset_val.scaler,
                                        batch_x,
                                        batch_y)
            loss = loss_fn(pred_scale.detach().cpu(),
                           true_scale.detach().cpu()
                           )
            if mid_scale != None:
                loss += loss_fn(mid_scale.detach().cpu(),
                                true_scale.detach().cpu()
                                )

            total_loss.append(loss)

        return np.average(total_loss)


    def train_and_saveModel(self, modelSavedName: str, hp: dict = None) -> tuple:
        """
        train for given epochs
        save best models
        return val_score, hp, model (for HPO)
        """
        model_savePath = os.path.join(self.args.checkpoints, modelSavedName)
        if not os.path.exists(model_savePath):
            os.makedirs(model_savePath)

        writer = SummaryWriter(
            f'event/run_{self.args.data}/{self.args.model_name}')

        earlyStopChecker = EarlyStopping(patience=self.args.patience,
                                         verbose=self.args.earlyStopVerbose)

        model_optim = self._select_optimizer()
        loss_fn = self._select_loss(self.args.loss)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        epoch_start = 0
        if self.args.resume:
            self.model, lr, epoch_start = load_model(self.model,
                                                     model_savePath,
                                                     model_name=self.args.data,
                                                     pred_len=self.args.pred_len
                                                     )

        # start epoch based training !!
        time_now = time.time()
        for epoch in range(epoch_start, self.args.train_epochs):

            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(self.train_loader):

                model_optim.zero_grad()

                true_scale, pred_scale, mid_scale = \
                    self._process_one_batch(self.dataset_train.scaler,
                                            batch_x,
                                            batch_y)
                # torch loss는 pred true순서임. true pred순서아님.
                loss_value = loss_fn(pred_scale, true_scale)
                if mid_scale != None:
                    loss_value += loss_fn(mid_scale, true_scale)

                train_loss.append(loss_value.item())

                if (i + 1) % self.print_per_iter == 0:
                    speed = (time.time() - time_now) / self.print_per_iter
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: "
                          f"{loss_value.item():.7f} | speed: {speed:.4f}s/iter")
                    time_now = time.time()

                if self.args.use_amp:  # use automatic mixed precision training
                    print('use amp')
                    scaler.scale(loss_value).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss_value.backward()
                    model_optim.step()

            # when finished one epoch
            train_loss = np.average(train_loss)
            val_loss = self.get_validation_loss(
                self.val_loader, loss_fn)
            print(
                f"Epoch: {epoch + 1} time: {time.time() - epoch_time:.1f}sec")
            print('--------start to validate-----------')
            print(f"Epoch: {epoch + 1}, Steps: {len(self.train_loader)} | "
                  f"Train Loss: {train_loss:.7f} valid Loss: {val_loss:.7f}")

            writer.add_scalar('train_loss', train_loss, global_step=epoch)
            writer.add_scalar('valid_loss', val_loss, global_step=epoch)

            earlyStopChecker(val_loss)
            if earlyStopChecker.counter == 0:  # improved !
                save_path = os.path.join(model_savePath,
                                         f'{self.args.data}_best.pt')
                save_model(epoch,
                           self.args.lr,
                           self.model,
                           save_path
                           )
            elif earlyStopChecker.early_stop:
                print("\n\n!!! Early stopping \n\n")
                break

            lr = adjust_learning_rate(model_optim, epoch + 1, self.args)
            assert lr == self.args.lr, 'adjust_learning_rate function is weird'

        # when finished all epoch
        save_path = os.path.join(model_savePath,
                                 f'{self.args.data}_last.pt')
        save_model(epoch,
                   self.args.lr,
                   self.model,
                   save_path
                   )

        return val_loss, hp


    def get_true_pred_of_testset(self, best_hp, modelSavedName) -> tuple:
        """test using saved best model
        !!! prediction results are rounded
        """
        self.model.eval()

        path = os.path.join(self.args.checkpoints, modelSavedName)
        best_model_path = f'{path}/checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        trues = []
        preds = []
        for batch_x, batch_y in self.test_loader:
            true_scale, pred_scale, _ = \
                self._process_one_batch(self.dataset_test.scaler,
                                        batch_x,
                                        batch_y)
            trues.append(true_scale.detach().cpu().numpy())
            preds.append(pred_scale.detach().cpu().numpy())

        y_tests = np.concatenate(trues, axis=0)
        pred_tests = np.clip(np.concatenate(preds, axis=0), 0, 1).round()

        return y_tests, pred_tests
