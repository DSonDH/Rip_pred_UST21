import os
import argparse
import torch
import itertools

from data_process import NIA_data_loader_csvOnly_YearSplit
from experiments.experiment_SARIMAX import Experiment_SARIMAX
from experiments.experiment_SVM import Experiment_SVM
from experiments.experiment_ML import Experiment_ML
from experiments.experiment_DL import Experiment_DL
from metrics.NIA_metrics import metric_classifier, metric_regressor, metric_all
from utils.tools import record_studyname_metrics
import pandas as pd


def parse_args(model: str,
               do_train: bool,
               gpu_idx: int,
               idx: int, 
               input_len: int,
               pred_len: int,
               input_dim: int,
               n_workers: int,
               epochs: int,
               bs: int,
               patience: int,
               lr: float
               ) -> None:
    study = 'NIA'
    NIA_work = 'ripcurrent_100p'  # data 전처리 meta file 저장이름 변경용
    root_path = f'./datasets/{study}/'
    year = 'allYear'  # data read할 csv파일
    port = 'AllPorts'
    fname = f'obs_qc'

    parser = argparse.ArgumentParser(description=f'{model} on {study} dataset')
    parser.add_argument('--model',
                        type=str,
                        required=False,
                        default=f'{model}_{study}_{port}',
                        help='model of the experiment')

    # -------  dataset settings --------------
    parser.add_argument('--NIA_work',
                        type=str,
                        required=False,
                        default=NIA_work,
                        help='work name of NIA')
    parser.add_argument('--data',
                        type=str,
                        required=False,
                        default=study,
                        help='name of dataset')
    parser.add_argument('--year',
                        type=str,
                        required=False,
                        default=year,
                        help='Dataset year')
    parser.add_argument('--port',
                        type=str,
                        required=False,
                        default=port,
                        help='name of port')
    parser.add_argument('--root_path',
                        type=str,
                        default=root_path,
                        help='root path of the data file')
    parser.add_argument('--data_path',
                        type=str,
                        default=fname,
                        help='location of the data file')
    parser.add_argument('--checkpoints',
                        type=str,
                        default=f'exp/{study}_checkpoints/',
                        help='location of model checkpoints')
    parser.add_argument('--inverse',
                        type=bool,
                        default=False,
                        help='denorm the output data')
    parser.add_argument('--embed',
                        type=str,
                        default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')

    # -------  device settings --------------
    parser.add_argument('--use_gpu',
                        type=bool,
                        default=True,
                        help='use gpu')
    parser.add_argument('--gpu',
                        type=int,
                        default=1,
                        help='gpu')
    parser.add_argument('--use_multi_gpu',
                        action='store_true',
                        help='use multiple gpus',
                        default=False)
    parser.add_argument('--devices',
                        type=str,
                        default=gpu_idx,
                        help='device ids of multile gpus')

    # -------  input/output length and 'number of feature' settings ------------
    parser.add_argument('--input_dim',
                        type=int,
                        default=input_dim,
                        help='number of input features')
    parser.add_argument('--input_len',
                        type=int,
                        default=input_len,
                        help='input seq length of encoder, look back window')
    parser.add_argument('--pred_len',
                        type=int,
                        default=pred_len,
                        help='prediction sequence length, horizon')
    parser.add_argument('--concat_len',
                        type=int,
                        default=0)
    parser.add_argument('--single_step',
                        type=int,
                        default=0)
    parser.add_argument('--single_step_output_One',
                        type=int,
                        default=0)
    parser.add_argument('--lastWeight',
                        type=float,
                        default=1.0)

    # -------  training settings --------------
    parser.add_argument('--cols',
                        type=str,
                        nargs='+',
                        help='file list')
    parser.add_argument('--num_workers',
                        type=int,
                        default=n_workers,
                        help='data loader num workers')
    parser.add_argument('--train_epochs',
                        type=int,
                        default=epochs,
                        help='train epochs')
    parser.add_argument('--batch_size',
                        type=int,
                        default=bs,
                        help='batch size of train input data')
    parser.add_argument('--patience',
                        type=int,
                        default=patience,
                        help='early stopping patience')
    parser.add_argument('--lr',
                        type=float,
                        default=lr,
                        help='optimizer learning rate')
    parser.add_argument('--loss',
                        type=str,
                        default='mae',
                        help='loss function')
    parser.add_argument('--lradj',
                        type=int,
                        default=1,
                        help='adjust learning rate')
    parser.add_argument('--use_amp',
                        action='store_true',
                        help='use automatic mixed precision training',
                        default=False)
    parser.add_argument('--save',
                        type=bool,
                        default=False,
                        help='save the output results')
    parser.add_argument('--model_name',
                        type=str,
                        default=f'{model}')
    parser.add_argument('--resume',
                        type=bool,
                        default=False)
    # only when you finished trainig
    parser.add_argument('--do_train',
                        type=bool,
                        default=do_train)

    # -------  model settings --------------
    parser.add_argument('--hidden-size',
                        default=1,
                        type=float,
                        help='hidden channel of module')
    parser.add_argument('--INN',
                        default=1,
                        type=int,
                        help='use INN or basic strategy')
    parser.add_argument('--kernel',
                        default=5,
                        type=int,
                        help='kernel size, 3, 5, 7')
    parser.add_argument('--dilation',
                        default=1,
                        type=int,
                        help='dilation')
    parser.add_argument('--window_size',
                        default=3,
                        type=int,
                        help='input size')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.5,
                        help='dropout')
    parser.add_argument('--positionalEcoding',
                        type=bool,
                        default=False)
    parser.add_argument('--groups',
                        type=int,
                        default=2)
    parser.add_argument('--levels',
                        type=int,
                        default=2)
    parser.add_argument('--stacks',
                        type=int,
                        default=2,
                        help='1 stack or 2 stacks')
    parser.add_argument('--num_decoder_layer',
                        type=int,
                        default=1)
    parser.add_argument('--RIN',
                        type=bool,
                        default=False)
    parser.add_argument('--decompose',
                        type=bool,
                        default=True)

    args = parser.parse_args()
    return args


def call_experiments_record_performances(model: str,
                                         do_train: bool,
                                         gpu_idx: int,
                                         itv: int,
                                         input_len: int,
                                         pred_len: int,
                                         input_dim: int,
                                         n_workers: int,
                                         epochs: int,
                                         bs: int,
                                         patience: int,
                                         lr: float
                                         ) -> None:
    """ do rip current prediction of 1h, 3h, 6h
        using classification fashion and regression fashion models
    args:
        model: model name
        do_train: do train or load trained model for testing
        gpu_idx: gpu index for trainig, testing
        itv: how many time point is in 1-hour? (default: 12)
        input_len: input sequence length
        pred_len: prediction sequence length
        input_dim: number of input feature
        n_workers: number of workers for preprocessing
        epochs: number of epochs for training
        bs: batch size
        patience: training patience
        lr: learning rate
    return: pd.dataframe recording experiment results
    """

    args = parse_args(
        model, do_train, gpu_idx, itv, input_len,
        pred_len, input_dim, n_workers, epochs,
        bs, patience, lr
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

    setting = '{}_{}_sl{}_pl{}_lr{}_bs{}_hid{}_s{}_l{}_dp{}_inv{}'.format(
        args.model, args.data, args.input_len, args.pred_len, args.lr,
        args.batch_size, args.hidden_size, args.stacks,
        args.levels, args.dropout, args.inverse)

    # experiment major types:
    # 1. Traditional (SARIMAX, SVM)
    # 2. Machine Learning (RF, XGB)
    # 3. Deep Learning (MLP famaily, RNN family, 1DCNN family)
    if os.path.exists('./results/Results.csv'):
        df = pd.read_csv('./results/Results.csv')
    else:
        df = pd.DataFrame(columns=['study_name', 'acc', 'f1', 'TP', 'FP', 'FN',
                                   'TN', 'mae', 'mse', 'rmse', 'mape', 'mspe',
                                   'corr']
                          )

    if args.model_name == 'SARIMAX':
        DatasetClass = NIA_data_loader_csvOnly_YearSplit.Dataset_NIA_class
        data_set_test = DatasetClass(args=args, flag='test', is_2d=False)

        assert args.pred_len == itv * 2, 'pred length of SARIMAX should be 24'
        # SARIMAX는 training과정이 없으며, 언제나 testset을 활용함
        y_test_label, pred_test = Experiment_SARIMAX(
            data_set_test,
            pred_len=pred_len,
            n_worker=20
        )

        y_test_org = data_set_test.scaler.inverse_transform(y_test_label)[:, :, 10]
        for i in [1, 2]: # time of interest metric
            metrics = metric_classifier(y_test_org[:, itv * i - 1],
                                        pred_test[:, itv * i - 1]
                                        )
            study_name = f'{args.model_name}_predH{i}_IL{args.input_len}_PL{args.pred_len}_clasf'
            df = record_studyname_metrics(df, study_name, metrics)
        
        # full time metric
        metrics_allRange = metric_all(y_test_org, pred_test)
        study_name = f'{args.model_name}_predH0~2_IL{args.input_len}_PL{args.pred_len}_regrs'
        df = record_studyname_metrics(df, study_name, metrics_allRange)
        df.to_csv('./results/Results.csv', index=False)

    elif args.model_name in ['SVM']:
        DatasetClass = NIA_data_loader_csvOnly_YearSplit.Dataset_NIA_class
        data_set_train = DatasetClass(args=args, flag='train', is_2d=True)
        data_set_val = DatasetClass(args=args, flag='val', is_2d=True)
        data_set_test = DatasetClass(args=args, flag='test', is_2d=True)

        assert args.pred_len == itv * 2, 'pred length of SARIMAX should be 24'
        
        y_test_label, pred_test = Experiment_SVM(
            data_set_train,
            data_set_val,
            data_set_test,
            pred_len=pred_len,
            n_worker=20
        )

        y_test_org = data_set_test.scaler.inverse_transform(y_test_label)[:, :, 10]
        for i in [1, 2]: # time of interest metric
            metrics = metric_classifier(y_test_org[:, itv * i - 1],
                                        pred_test[:, itv * i - 1]
                                        )
            study_name = f'{args.model_name}_predH{i}_IL{args.input_len}_PL{args.pred_len}_clasf'
            df = record_studyname_metrics(df, study_name, metrics)
        
        # full time metric
        metrics_allRange = metric_all(y_test_org, pred_test)
        study_name = f'{args.model_name}_predH0~2_IL{args.input_len}_PL{args.pred_len}_regrs'
        df = record_studyname_metrics(df, study_name, metrics_allRange)
        df.to_csv('./results/Results.csv', index=False)


    elif args.model_name in ['RF', 'XGB']:
        """
        TODO: result should return 3 experiments
            1. each prediction time model
            2. all prediction time at once prediciton model
            3. seq2seq model (all range at once)
        then I'm going to compare three results and choose to report at the paper
        => ML로 단일시간 예측하는게 좋은지, seq2seq도 좋은지 얘기할것임
        """
        # TODO: tr, val, te 모드 정보가 들어가는 지 확인
        # TODO: 위 주석에 언급한 3개 모드 중 어떤걸로 돌리는지 옵션으로 들어가야 함

        # TODO: 1, 3, 6 예측시간에 대해서 pred_len을 달리 해야할까?
        # 입력자료 길이는 어떻게 고정할까 ... ? HP로 두고 tuning할까 ?

        y_test_label1, pred_test1 = Experiment_ML(
            setting, 'mode1')  # seq2scalar
        y_test_label3, pred_test3 = Experiment_ML(setting, 'mode2')  # seq2vec
        y_test_label6, pred_test6 = Experiment_ML(setting, 'mode3')  # seq2seq

        # calc metrics
        metrics1 = metric_classifier(y_test_label1, pred_test1)
        metrics2_1 = metric_classifier(y_test_label2[:, 0], pred_test2[:, 0])
        metrics2_2 = metric_classifier(y_test_label2[:, 1], pred_test2[:, 1])
        metrics2_3 = metric_classifier(y_test_label2[:, 2], pred_test2[:, 2])
        metrics3 = metric_all(y_test_label3, pred_test3)

    else:  # DL models
        """
        TODO: result should return 3 experiments
            1. each prediction time model
            2. all prediction time at once prediciton model
            3. seq2seq model (all range at once)
        then I'm going to compare three results and choose to report at the paper
        => DL로 단일시간 예측하는게 좋은지, seq2seq이 좋은지 얘기할것임
        """
        # TODO: tr, val, te 모드 정보가 들어가는 지 확인
        # TODO: 위 주석에 언급한 3개 모드 중 어떤걸로 돌리는지 옵션으로 들어가야 함

        # TODO: 1, 3, 6 예측시간에 대해서 pred_len을 달리 해야할까?
        # TODO: scinet 같은 경우는 input_len, pred_len이 2의 제곱이 되야하므로
        # 따로 처리하는 코드 필요

        # 입력자료 길이는 어떻게 고정할까 ... ? HP로 두고 tuning할까 ?

        y_test_label, pred_test = Experiment_DL(setting)

        # calc metrics
        metrics1 = metric_classifier(y_test_label1, pred_test1)
        metrics2_1 = metric_classifier(y_test_label2[:, 0], pred_test2[:, 0])
        metrics2_2 = metric_classifier(y_test_label2[:, 1], pred_test2[:, 1])
        metrics2_3 = metric_classifier(y_test_label2[:, 2], pred_test2[:, 2])
        metrics3 = metric_all(y_test_label3, pred_test3)


if __name__ == '__main__':
    # ===================================================
    # configs usually changed

    model = 'SVM'  # FIXME:
    # SARIMAX, SVM, RF, XGB, MLPvanilla, SimpleLinear, LightTS,
    # Simple1DCNN, SCINET, LSTM, Transformer, Informer
    do_train = True  # FIXME:
    gpu_idx = '1'  # FIXME:

    # data setting. !! 5분 간격이므로 1시간에 12개 존재함
    itv = 12

    # pred_len보다 2배는 길게 input_len 설정하는 듯.
    # 6시간 예측이면 12시간 input넣어줘야 하는데, 길이기 길면 길수록 결측도 많아지므로
    # 샘플이 급격히 줄거나 class imbalance가 증가할 수도 있음 이에 따른 성능 하락도 고려해야함
    # 그리고 모델 complexity에 따라서 필요한 input sequence가 달라진다고 하니깐.
    # 최종 best 모델로 결론 낼 때에 맞는 input_len을 제시하면 될듯
    input_len = [itv * i for i in [2, 4]]  # FIXME:
    pred_len = [itv * i for i in [2]]  # FIXME:
    # pred_len = [itv * i for i in [1, 2]]  # FIXME:
    input_dim = 11  # FIXME: n_feature. site정보인 onehot vector는 넣지 않기로 함

    # trainig setting
    epochs = 100  # FIXME:
    patience = 10  # FIXME:
    batchSize = 32
    learningRate = 0.001
    n_workers = 10
    # ===================================================

    for input_len_tmp, pred_len_tmp in list(itertools.product(input_len, pred_len)):
        # print(input_len_tmp, pred_len_tmp)
        call_experiments_record_performances(model, do_train, gpu_idx,
                                             itv, input_len_tmp, pred_len_tmp,
                                             input_dim, n_workers, epochs,
                                             batchSize, patience, learningRate
                                             )
