import os
import argparse
import torch
import itertools
import json
import shutil
from datetime import datetime

from data_process import NIA_data_loader_csvOnly_YearSplit
from experiments.experiment_SARIMAX import Experiment_SARIMAX
from experiments.experiment_SVM import Experiment_SVM
from experiments.experiment_ML import Experiment_ML
from experiments.experiment_DL import Experiment_DL
from utils.metrics.NIA_metrics import metric_classifier, metric_regressor, metric_all
from utils.tools import record_studyname_metrics
from models.Tuning_configs import model_tuning_dict

import numpy as np
import pandas as pd


def parse_args(model: str,
               do_train: bool,
               gpu_idx: int,
               input_len: int,
               pred_len: int,
               tois: int,
               n_workers: int,
               epochs: int,
               bs: int,
               patience: int,
               lr: float,
               date_prefix: str,
               time_prefix: str,
               ) -> None:
    study = 'NIA'
    NIA_work = 'ripcurrent_100p'  # data 전처리 meta file 저장이름 변경용
    root_path = f'./datasets/{study}/'
    year = 'allYear'  # data read할 csv파일
    port = 'AllPorts'
    fname = f'obs_qc'

    parser = argparse.ArgumentParser(description=f'{model} on {study} dataset')
    parser.add_argument('--model',
                        type=str, required=False,
                        default=f'{model}_{study}_{port}',
                        help='model of the experiment')

    # -------  dataset settings --------------
    parser.add_argument('--NIA_work',
                        type=str, required=False, default=NIA_work,
                        help='work name of NIA')
    parser.add_argument('--data',
                        type=str, required=False, default=study,
                        help='name of dataset')
    parser.add_argument('--year',
                        type=str, required=False, default=year,
                        help='Dataset year')
    parser.add_argument('--port',
                        type=str, required=False, default=port,
                        help='name of port')
    parser.add_argument('--root_path',
                        type=str, default=root_path,
                        help='root path of the data file')
    parser.add_argument('--data_path',
                        type=str, default=fname,
                        help='location of the data file')
    parser.add_argument('--ckpt_path',
                        type=str, default=f'results/exp/{study}_checkpoints',
                        help='location of model checkpoints')
    parser.add_argument('--embed',
                        type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')

    # -------  device settings --------------
    parser.add_argument('--use_gpu',
                        type=bool, default=True,
                        help='use gpu')
    parser.add_argument('--gpu',
                        type=int, default=1,
                        help='gpu')
    parser.add_argument('--use_multi_gpu',
                        action='store_true', help='use multiple gpus',
                        default=False)
    parser.add_argument('--devices',
                        type=str, default=gpu_idx,
                        help='device ids of multile gpus')

    # -------  input/output length and 'number of feature' settings ------------
    parser.add_argument('--itv',
                        type=int, default=12,
                        help='number of time point in one hour')
    parser.add_argument('--input_dim',
                        type=int, default=11,
                        help='number of input features, including rip label')
    parser.add_argument('--input_len',
                        type=int, default=input_len,
                        help='input seq length of encoder, look back window')
    parser.add_argument('--pred_len',
                        type=int, default=pred_len,
                        help='prediction sequence length')
    parser.add_argument('--tois',
                        type=int, default=tois,
                        help='time of interest to evaluate')
    # for SCINet
    parser.add_argument('--concat_len',
                        type=int, default=0)
    parser.add_argument('--single_step',
                        type=int, default=0)
    parser.add_argument('--single_step_output_One',
                        type=int, default=0)

    # -------  training settings --------------
    parser.add_argument('--cols',
                        type=str, nargs='+',
                        help='file list')
    parser.add_argument('--num_workers',
                        type=int, default=n_workers,
                        help='data loader num workers')
    parser.add_argument('--train_epochs',
                        type=int, default=epochs,
                        help='train epochs')
    parser.add_argument('--batch_size',
                        type=int, default=bs,
                        help='batch size of train input data')
    parser.add_argument('--patience',
                        type=int, default=patience,
                        help='early stopping patience')
    parser.add_argument('--earlyStopVerbose',
                        type=bool, default=True,
                        help='choose to show early stop message or not')
    parser.add_argument('--lr',
                        type=float, default=lr,
                        help='optimizer learning rate')
    parser.add_argument('--loss',
                        type=str, default='mae',
                        help='loss function')
    parser.add_argument('--lradj',
                        type=int, default=1,
                        help='adjust learning rate')
    parser.add_argument('--use_amp',
                        action='store_true',
                        help='use automatic mixed precision training',
                        default=False)
    parser.add_argument('--save',
                        type=bool, default=False,
                        help='save the output results')
    parser.add_argument('--model_name',
                        type=str, default=f'{model}')
    parser.add_argument('--resume',
                        type=bool, default=False)
    # only when you finished trainig
    parser.add_argument('--do_train',
                        type=bool, default=do_train)
    parser.add_argument('--date_prefix',
                        type=str, default=date_prefix)
    parser.add_argument('--time_prefix',
                        type=str, default=time_prefix)

    args = parser.parse_args()
    return args


def call_experiments_record_performances(model: str,
                                         do_train: bool,
                                         gpu_idx: int,
                                         input_len: int,
                                         pred_len: int,
                                         tois: int,
                                         n_workers: int,
                                         epochs: int,
                                         bs: int,
                                         patience: int,
                                         lr: float,
                                         date_prefix: str,
                                         time_prefix: str,
                                         ) -> None:
    """ do rip current prediction of 1h, 3h, 6h
        using classification fashion and regression fashion models
    args:
        model: model name
        do_train: do train or load trained model for testing
        gpu_idx: gpu index for trainig, testing
        input_len: input sequence length
        pred_len: prediction sequence length
        tois: time of interests for evaluation
        n_workers: number of workers for preprocessing
        epochs: number of epochs for training
        bs: batch size
        patience: training patience
        lr: learning rate
    return: pd.dataframe recording experiment results
    """
    assert model in ['SARIMAX', 'SVM', 'RF', 'XGB', 'MLPvanilla', 'Simple1DCNN',
                     '1DCNN', 'LSTM', 'Transformer', 'LTSF-Linear', 'LightTS',
                     'SCINet', 'Informer']

    args = parse_args(
        model,
        do_train,
        gpu_idx,
        input_len,
        pred_len,
        tois,
        n_workers,
        epochs,
        bs,
        patience,
        lr,
        date_prefix,
        time_prefix,
    )

    print('\n\n')
    print('='*80)
    print('|', ' '*24, f' ***[ {args.port} ]*** Start !', ' '*23, '|')
    print('='*80)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and not args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        torch.cuda.set_device(args.gpu)
    elif args.use_gpu and not args.use_multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=[1, 2, 3])

    print('Args in experiment:')
    print(args, '\n\n')

    torch.manual_seed(4321)  # reproducible
    torch.cuda.manual_seed_all(4321)
    torch.backends.cudnn.benchmark = False
    # Can change it to False --> default: False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    # experiment major types:
    # 1. Traditional (SARIMAX, SVM)
    # 2. Machine Learning (RF, XGB)
    # 3. Deep Learning (MLP famaily, RNN family, 1DCNN family)
    if os.path.exists('./results/Results.csv'):
        df = pd.read_csv('./results/Results.csv')
    else:
        df = pd.DataFrame(columns=['time', 'study_name',
                                   'acc', 'f1', 'TP', 'FP', 'FN',
                                   'TN', 'mae', 'mse', 'rmse', 'mape', 'mspe',
                                   'corr']
                          )

    DatasetClass = NIA_data_loader_csvOnly_YearSplit.Dataset_NIA_class

    if args.model_name == 'SARIMAX':
        data_set_test = DatasetClass(args=args, flag='test', is2d=False)

        assert args.pred_len == args.itv * 2, \
            'SARIMAX should have 2hour prediction length'
        # 1시간 예측만 따로하지 않으므로. 1시간, 2시간 함께 예측하려면 2시간 필요

        # SARIMAX는 training과정이 없으며, 언제나 testset을 활용함
        y_test, pred_test = Experiment_SARIMAX(
            data_set_test,
            pred_len=args.pred_len,
            n_worker=20
        )

        for toi in args.tois:  # time of interest metric
            metrics = metric_classifier(y_test[:, args.itv * toi - 1],
                                        pred_test[:, args.itv * toi - 1]
                                        )
            study_name = f'{args.model_name}_predH{toi}_IL{args.input_len}_'\
                         f'PL{args.pred_len}_clasf'
            df = record_studyname_metrics(df, args, study_name, metrics)

        # full time metric
        metrics_allRange = metric_all(y_test, pred_test)
        study_name = f'{args.model_name}_predH0~2_IL{args.input_len}_'\
                     f'PL{args.pred_len}_regrs'
        df = record_studyname_metrics(df, args, study_name, metrics_allRange)
        df.to_csv('./results/Results.csv', index=False)

    elif args.model_name in ['SVM']:
        # SVM은 한번에 여러 시간 예측하는게 아닌, 각각 단일 예측이므로 tois loop없음
        data_set_train = DatasetClass(args=args, flag='train', is2d=True)
        data_set_val = DatasetClass(args=args, flag='val', is2d=True)
        data_set_test = DatasetClass(args=args, flag='test', is2d=True)

        y_test, pred_test = Experiment_SVM(
            data_set_train,
            data_set_val,
            data_set_test,
            pred_len=args.pred_len,
            n_worker=10
        )

        metrics = metric_classifier(y_test, pred_test)
        study_name = f'{args.model_name}_predH{args.pred_len // args.itv}_'\
            f'IL{args.input_len}_clasf'
        df = record_studyname_metrics(df, args, study_name, metrics)
        df.to_csv('./results/Results_SVM.csv', index=False)
        # SVM은 너무 느려서 따로 파일 만듦

    elif args.model_name in ['RF', 'XGB']:
        """code block for Random Forest, Extra Gradient Boosting
        pred len 2일때 : from multi (pred1, pred2, pred1~2)
                        from single (pred2) 가능
        pred len 1일때 : from single (pred1) 가능
        """
        data_set_train = DatasetClass(args=args, flag='train', is2d=True)
        data_set_val = DatasetClass(args=args, flag='val', is2d=True)
        data_set_test = DatasetClass(args=args, flag='test', is2d=True)

        data_set_train_3d = DatasetClass(args=args, flag='train', is2d=True)
        data_set_val_3d = DatasetClass(args=args, flag='val', is2d=True)
        data_set_test_3d = DatasetClass(args=args, flag='test', is2d=True)

        if args.pred_len == args.itv * 2:
            # seq2seq mode
            y_test, pred_test = Experiment_ML(data_set_train_3d,
                                              data_set_val_3d,
                                              data_set_test_3d,
                                              pred_len=args.pred_len,
                                              n_worker=20,
                                              mode='seq2seq',
                                              args=args
                                              )

            # time of interest select and calculate
            for toi in args.tois:
                metrics = metric_classifier(y_test[:, args.itv * toi - 1],
                                            pred_test[:, args.itv * toi - 1]
                                            )
                study_name = f'{args.model_name}_predH{toi}_IL{args.input_len}_'\
                    f'PL{args.pred_len}_clasf'
                df = record_studyname_metrics(df, args, study_name, metrics)

            # full time metric
            metrics_allRange = metric_all(y_test, pred_test)
            study_name = f'{args.model_name}_predH0~2_IL{args.input_len}_'\
                f'PL{args.pred_len}_regrs'
            df = record_studyname_metrics(
                df, args, study_name, metrics_allRange)

            # single mode metric for 2hour prediction
            y_test_2h, pred_test_2h = Experiment_ML(data_set_train,
                                                    data_set_val,
                                                    data_set_test,
                                                    pred_len=args.pred_len,
                                                    n_worker=20,
                                                    mode='single',
                                                    args=args
                                                    )
            metrics = metric_classifier(y_test_2h, pred_test_2h)
            study_name = f'{args.model_name}_predH{args.pred_len // args.itv}_'\
                f'IL{args.input_len}_clasf'
            df = record_studyname_metrics(df, study_name, metrics)

        else:
            assert args.pred_len // args.itv == 1, 'In ML, invalid pred_length !!'
            y_test_1h, pred_test_1h = Experiment_ML(data_set_train,
                                                    data_set_val,
                                                    data_set_test,
                                                    pred_len=args.pred_len,
                                                    n_worker=20,
                                                    mode='single',
                                                    args=args)
            metrics = metric_classifier(y_test_1h, pred_test_1h)
            study_name = f'{args.model_name}_predH{args.pred_len // args.itv}_'\
                f'IL{args.input_len}_clasf'
            df = record_studyname_metrics(df, study_name, metrics)

        df.to_csv('./results/Results.csv', index=False)

    else:  # DL models
        # TODO: tuning별 성능정리(train, val log, test score)보면 재밌으려나...?
        # TODO: 1DCNN, LSTM, Transformer, LTSF-Linear, LightTS, SCINet 순서로 구현
        # TODO: scinet은 input_len, pred_len이 2의 제곱이 되야하므로 입력자료 따로 처리?

        assert args.pred_len == args.itv * 2, \
            'DL models should have 2hour prediction length'

        # output directory setting
        dirName = f'{args.ckpt_path}/{args.model_name}_{args.date_prefix}'
        if os.path.exists(dirName):
            dirName = f'{dirName}/{args.time_prefix}'
        if not os.path.exists(dirName):
            os.mkdir(dirName)

        # training with tuning
        if args.do_train:
            """do hyperParameter Optimization using grid search"""
            print(f'{args.model_name}: Start Training with tuning !!')

            default_dict = {
                'scale_pos_weight': 2,
                'objective': 'logistic',
                'early_stopping_rounds': 15,
                'seed': 1,
                'verbosity': 0
            }

            tuning_dict = model_tuning_dict(args)

            # start learning and tuning
            best_loss = np.inf
            hyperparameters = list(itertools.product(*tuning_dict.values()))
            for i, hp in enumerate(hyperparameters):
                print(f'--- {args.model}: HPO '
                      f'processing {i / len(hyperparameters) * 100:.1f}% ---')

                tuning_dict_tmp = {k: v for k, v in zip(tuning_dict.keys(), hp)}
                tuning_dict_tmp.update(default_dict)

                # tuning trial들 임시 저장
                modelSaveDir = f'{dirName}/HPO_trial{i}'
                if not os.path.exists(modelSaveDir):
                    os.mkdir(modelSaveDir)

                print(f'\n         ======  Now Tuning with trial {i} ======')
                print(f'{tuning_dict_tmp}')
                print('         ===========================================\n')

                DL_experiment = Experiment_DL(args, tuning_dict_tmp)

                val_loss = DL_experiment.train_and_saveModel(modelSaveDir) # mae
                if val_loss <= best_loss:  # lower the loss, the better model
                    best_idx = i
                    best_loss = val_loss
                    best_hp = tuning_dict_tmp

            # after all tuning done
            print(f'\n\n***** best index is {best_idx} !!! *****')
            for i in range(len(hyperparameters)):
                if i != best_idx:
                    # remove trial histories
                    shutil.rmtree(f'{dirName}/HPO_trial{i}')
                else:
                    # keep best model
                    os.system(f'mv {dirName}/HPO_trial{i}/* {dirName}')
                    shutil.rmtree(f'{dirName}/HPO_trial{i}')

                    # save best hyper-parameters as json
                    #FIXME: 학습 다 끝나고 TypeError: Object of type int64 is not JSON serializable
                    json_saveName = f'{dirName}/bestHyperParams.json'
                    with open(json_saveName, "w") as outfile:
                        json.dump(best_hp, outfile, indent=4)

        # do test
        print(f'{args.model}: Start Testing')

        with open(f'{dirName}/bestHyperParams.json', "rb") as f:
            best_hp = json.load(f)

        DL_experiment = Experiment_DL(args, best_hp)

        savedName = f'{dirName}/{args.model}_il{args.input_len}'\
                    f'_pl{args.pred_len}'

        y_test, pred_test = DL_experiment.get_testResults(savedName)

        # get accuracy
        for toi in args.tois:
            metrics = metric_classifier(y_test[:, args.itv * toi - 1],
                                        pred_test[:, args.itv * toi - 1]
                                        )
            study_name = f'{args.model_name}_predH{toi}_IL{args.input_len}_'\
                         f'PL{args.pred_len}_clasf'
            df = record_studyname_metrics(df, args, study_name, metrics)

        metrics_allRange = metric_all(y_test, pred_test)
        study_name = f'{args.model_name}_predH0~2_IL{args.input_len}_'\
                     f'PL{args.pred_len}_regrs'
        df = record_studyname_metrics(df, args, study_name, metrics_allRange)

        df.to_csv('./results/Results.csv', index=False)


if __name__ == '__main__':
    # ===================================================
    # TODO:FIXME: 얘네도 tuning_configs안에 넣기 configs usually changed

    # models = ['SARIMAX']  # FIXME:
    # models = ['RF', 'XGB']  # FIXME:
    # 'MLPvanilla', 
    models = ['Simple1DCNN', 'LSTM', 'Transformer',
              'LTSF-Linear', 'LightTS', 'SCINet', 'Informer']  # FIXME:
    for model in models:
        # SARIMAX, SVM, ML(RF, XGB),
        # DL (MLPvanilla, Simple1DCNN, 1DCNN, LSTM, Transformer, LTSF-Linear,
        # LightTS, SCINet, Informer)
        do_train = True  # FIXME:

        if do_train:
            # 한 날짜가 오래 돌아가는 작업 끝까지 유지돼야함
            # 같은 날에 여러번 돌리는 작업은 시분초 단위로 독립적으로 생성되도록 함
            date_prefix = datetime.today().strftime('%Y%m%d')
            time_prefix = datetime.today().strftime('%H%M%S')
        else:
            # test할 때의 시간은 train시간이랑 다를 것이므로, 기존 날짜를 입력해야 함
            date_prefix = '20230516'  # FIXME:
            time_prefix = '100542'  # FIXME:

        gpu_idx = '1'  # FIXME:

        # pred_len보다 2배는 길게 input_len 설정하는 듯.
        # 6시간 예측이면 12시간 input넣어줘야 하는데, 길이가 길면 길수록 결측도 많아지므로
        # 샘플이 급격히 준다 (실제로 그러함). 그리고 class imbalance가 증가할 수도 있음
        # 이에 따른 성능 하락도 고려해야함
        # 그리고 모델 complexity에 따라서 필요한 input sequence가 달라진다고 하니깐.
        # 최종 best 모델로 결론 낼 때에 맞는 input_len을 제시하면 될듯

        #!!! 아래 네줄은 아예 틀린거 아니면 바꾸지 말기
        # dataloader까지도 영향주는 파라미터임
        itv = 12  # how many time points are exist in one hour?
        input_lengths = [itv * i for i in [2, 4]]
        pred_lengths = [itv * i for i in [2, 1]] \
            if model in ['RF', 'XGB'] else [itv * 2]
        tois = [1, 2]  # prediction hour which we are interested in

        # DL trainig setting
        epochs = 2  # FIXME:
        patience = 10  # FIXME:
        batchSize = 32
        learningRate = 0.001
        n_workers = 10
        # ===================================================

        for input_len_tmp, pred_len_tmp in list(itertools.product(
                input_lengths, pred_lengths)):
            # print(input_len_tmp, pred_len_tmp)
            call_experiments_record_performances(model,
                                                 do_train,
                                                 gpu_idx,
                                                 input_len_tmp,
                                                 pred_len_tmp,
                                                 tois,
                                                 n_workers,
                                                 epochs,
                                                 batchSize,
                                                 patience,
                                                 learningRate,
                                                 date_prefix,
                                                 time_prefix,
                                                 )
            
