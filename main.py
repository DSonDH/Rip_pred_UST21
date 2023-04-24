import argparse
import torch
from experiments.experiment_DL import Experiment_DL
from experiments.experiment_SARIMAX import Experiment_SARIMAX
from experiments.experiment_ML import Experiment_ML

from metrics.NIA_metrics import metric_classifier, metric_regressor, metric_all

""" rip current prediction of 1h, 3h, 6h
using classification fashion and regression fashion models
"""
# ===================================================
# configs usually changed

model = 'SARIMAX'
# SARIMAX
# RF
# XGB
# MLPvanilla
# SimpleLinear
# LightTS
# Simple1DCNN
# SCINET
# LSTM
# Transformer
# Informer

do_train = True #FIXME:
gpu_idx = '1' #FIXME:

# model setting
input_len = 30 #FIXME:
pred_len = 36 #FIXME:
itv = 5  # timepoint 간격이 얼마인지에 따라서 인덱싱 달리

# TODO: 11 + 5에서 onehot 효과 없음이 보여지면 11개 feature만 쓰도록.
input_dim = 11 + 5  # n_feature + 5 one-hot

# trainig setting
n_workers = 10
epochs = 3
bs = 32
patience = 30
lr = 0.001
# is_new_test = False
# ===================================================
study = 'NIA'
NIA_work = 'ripcurrent_100p'  # data 전처리 meta file 저장이름 변경용
root_path = f'./datasets/{study}/'
year = 'allYear'  # data read할 csv파일
port_list = ['AllPorts']
fname = f'obs_qc'


for port in port_list:
    print('\n\n')
    print('='*80)
    print('|',' '*24, f' ***[ {port} ]*** Start !',' '*23,'|')
    print('='*80)

    parser = argparse.ArgumentParser(description=f'{model} on {study} dataset')
    parser.add_argument('--model', type=str, required=False, default=f'{model}_{study}_{port}', help='model of the experiment')
    ### -------  dataset settings --------------
    parser.add_argument('--NIA_work', type=str, required=False, default=NIA_work, help='work name of NIA')
    parser.add_argument('--data', type=str, required=False, default=study, help='name of dataset')
    parser.add_argument('--year', type=str, required=False, default=year, help='Dataset year')
    # parser.add_argument('--is_new_test', type=bool, required=False, default=is_new_test, help='need to make new test save file?')
    parser.add_argument('--itv', type=int, required=False, default=itv, help='name of dataset')
    parser.add_argument('--port', type=str, required=False, default=port, help='name of port')
    parser.add_argument('--root_path', type=str, default=root_path, help='root path of the data file')
    parser.add_argument('--data_path', type=str, default=fname, help='location of the data file')
    parser.add_argument('--checkpoints', type=str, default=f'exp/{study}_checkpoints/', help='location of model checkpoints')
    parser.add_argument('--inverse', type=bool, default =False, help='denorm the output data')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')

    ### -------  device settings --------------
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=1, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default=gpu_idx,help='device ids of multile gpus')

        ### -------  input/output length and 'number of feature' settings --------------                                                                            
    parser.add_argument('--in_dim', type=int, default=input_dim, help='number of input features')
    parser.add_argument('--input_len', type=int, default=input_len, help='input sequence length of model encoder, look back window')
    # parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')  # input, label곂치는걸 원치 않으므로 안씀
    parser.add_argument('--pred_len', type=int, default=pred_len, help='prediction sequence length, horizon')
    parser.add_argument('--concat_len', type=int, default=0)
    # parser.add_argument('--scale', type=bool, default=True)
    parser.add_argument('--single_step', type=int, default=0)
    parser.add_argument('--single_step_output_One', type=int, default=0)
    parser.add_argument('--lastWeight', type=float, default=1.0)

    ### -------  training settings --------------  
    parser.add_argument('--cols', type=str, nargs='+', help='file list')
    parser.add_argument('--num_workers', type=int, default=n_workers, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=epochs, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=bs, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=patience, help='early stopping patience')
    parser.add_argument('--lr', type=float, default=lr, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='mae',help='loss function')
    parser.add_argument('--lradj', type=int, default=1, help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--save', type=bool, default =False, help='save the output results')
    parser.add_argument('--model_name', type=str, default=f'{model}')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--do_train', type=bool, default=do_train)  # only when you finished trainig

    ### -------  model settings --------------  
    parser.add_argument('--hidden-size', default=1, type=float, help='hidden channel of module')
    parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
    parser.add_argument('--kernel', default=5, type=int, help='kernel size, 3, 5, 7')
    parser.add_argument('--dilation', default=1, type=int, help='dilation')
    parser.add_argument('--window_size', default=3, type=int, help='input size')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--positionalEcoding', type=bool, default=False)
    parser.add_argument('--groups', type=int, default=2)
    parser.add_argument('--levels', type=int, default=2)
    parser.add_argument('--stacks', type=int, default=2, help='1 stack or 2 stacks')
    parser.add_argument('--num_decoder_layer', type=int, default=1)
    parser.add_argument('--RIN', type=bool, default=False)
    parser.add_argument('--decompose', type=bool, default=True)

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and not args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        torch.cuda.set_device(args.gpu)
    elif args.use_gpu and not args.use_multi_gpu:
        pass
        # model= nn.DataParallel(model,device_ids = [1, 3])
    print('Args in experiment:')
    print(args, '\n\n')

    torch.manual_seed(4321)  # reproducible
    torch.cuda.manual_seed_all(4321)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
    torch.backends.cudnn.enabled = True

    setting = '{}_{}_sl{}_pl{}_lr{}_bs{}_hid{}_s{}_l{}_dp{}_inv{}'.format(
                    args.model,args.data, args.input_len, args.pred_len,args.lr,
                    args.batch_size,args.hidden_size,args.stacks, 
                    args.levels,args.dropout,args.inverse)

    
    #FIXME: 최종 save table 형식, 파일이 뭐가 되야지 연구하기 편할까?
    if args.model_name == 'SARIMAX':
        from data_process import NIA_data_loader_csvOnly_YearSplit
        DatasetClass = NIA_data_loader_csvOnly_YearSplit.Dataset_NIA

        data_set = DatasetClass(
            root_path = args.root_path,
            NIA_work = args.NIA_work,
            data = args.data,
            port = args.port,
            data_path = args.data_path,
            size = [args.input_len, args.pred_len],
            args = args
        )
        
        assert pred_len >= 35, \
            'designated pred length is shorter than expected output length'

        # SARIMAX는 training과정이 없으며, 언제나 testset을 활용함
        y_test_label, pred_test = Experiment_SARIMAX(
                                      data_set, 
                                      pred_style=pred_len,
                                      pred_len=pred_len,
                                      n_worker=201
                                    )

        # 원하는 시간 뽑아서 성능 계산
        metrics2_1 = metric_classifier(y_test_label[:, 5], pred_test[:, 5])
        metrics2_3 = metric_classifier(y_test_label[:, 17], pred_test[:, 17])
        metrics2_6 = metric_classifier(y_test_label[:, 35], pred_test[:, 35])

        # 전체 시간의 성능 계산
        metrics3 = metric_all(y_test_label3, pred_test3)

    elif args.model_name in ['RF', 'XGB']:        
        """
        TODO: result should return 3 experiments
            1. each prediction time model
            2. all prediction time at once prediciton model
            3. seq2seq model (all range at once)
        then I'm going to compare three results and choose to report at the paper
        => ML로 단일시간 예측하는게 좋은지, seq2seq도 좋은지 얘기할것임
        """
        #TODO: tr, val, te 모드 정보가 들어가는 지 확인
        #TODO: 위 주석에 언급한 3개 모드 중 어떤걸로 돌리는지 옵션으로 들어가야 함

        #TODO: 1, 3, 6 예측시간에 대해서 pred_len을 달리 해야할까?
        #입력자료 길이는 어떻게 고정할까 ... ? HP로 두고 tuning할까 ?


        y_test_label1, pred_test1 = Experiment_ML(setting, 'mode1')  # seq2scalar
        y_test_label3, pred_test3 = Experiment_ML(setting, 'mode2')  # seq2vec
        y_test_label6, pred_test6 = Experiment_ML(setting, 'mode3')  # seq2seq
        
        # calc metrics
        metrics1 = metric_classifier(y_test_label1, pred_test1)

        metrics2_1 = metric_classifier(y_test_label2[:, 0], pred_test2[:, 0])
        metrics2_2 = metric_classifier(y_test_label2[:, 1], pred_test2[:, 1])
        metrics2_3 = metric_classifier(y_test_label2[:, 2], pred_test2[:, 2])

        metrics3 = metric_all(y_test_label3, pred_test3)


    else: # DL models
        """
        TODO: result should return 3 experiments
            1. each prediction time model
            2. all prediction time at once prediciton model
            3. seq2seq model (all range at once)
        then I'm going to compare three results and choose to report at the paper
        => DL로 단일시간 예측하는게 좋은지, seq2seq이 좋은지 얘기할것임
        """
        #TODO: tr, val, te 모드 정보가 들어가는 지 확인
        #TODO: 위 주석에 언급한 3개 모드 중 어떤걸로 돌리는지 옵션으로 들어가야 함

        #TODO: 1, 3, 6 예측시간에 대해서 pred_len을 달리 해야할까?
        #TODO: scinet 같은 경우는 input_len, pred_len이 2의 제곱이 되야하므로 
        # 따로 처리하는 코드 필요

        #입력자료 길이는 어떻게 고정할까 ... ? HP로 두고 tuning할까 ?
        
        y_test_label, pred_test = Experiment_DL(setting)
        
        # calc metrics
        metrics1 = metric_classifier(y_test_label1, pred_test1)

        metrics2_1 = metric_classifier(y_test_label2[:, 0], pred_test2[:, 0])
        metrics2_2 = metric_classifier(y_test_label2[:, 1], pred_test2[:, 1])
        metrics2_3 = metric_classifier(y_test_label2[:, 2], pred_test2[:, 2])

        metrics3 = metric_all(y_test_label3, pred_test3)

    #FIXME: 최종 save table 형식, 파일이 뭐가 되야지 연구하기 편할까?


def print_performance(metrics: dict) -> None:
    print('*'*41)
    print(f'Final test metrics of {args.model_name}:\n')
    for key in metrics:
        print(f"{key}: {metrics[key]}\n")
    print('*'*41)