import argparse
import torch

# from torch.utils.tensorboard import SummaryWriter
from experiments.experiment import Experiment

# ===================================================
# configs for pipeline setting

study = 'NIA'  # KHOA  #FIXME:
NIA_work = 'ripcurrent_100p'  #FIXME: meta file name 변경용
root_path = f'./datasets/{study}/'
nia_csv_base = True # False : json file 내 날짜로 instance 생성
year = 'allYear'  # data read할 csv파일

model = 'DNN'

test_mode = False #FIXME: Test mode면 training 진행 안됨
gpu_idx = '0' #FIXME:
seq_len = 32
pred_len = 16

n_workers = 10
epochs = 200
bs = 32
patience = 30
lr = 0.001


is_new_test = False

itv = 5  # timepoint 간격이 얼마인지에 따라서 인덱싱 달리

# TODO: ROI 리스트를 config로 넘겨서 onehot vector (output길이 자르는거까지) 길이 정보 제공
in_dim = 11+5  # n_feature + 5 one-hot

# ===================================================

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
    parser.add_argument('--nia_csv_base', type=bool, required=False, default=nia_csv_base, help='gen data from csv or json')
    parser.add_argument('--is_new_test', type=bool, required=False, default=is_new_test, help='need to make new test save file?')
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
    parser.add_argument('--in_dim', type=int, default=in_dim, help='number of input features')
    parser.add_argument('--seq_len', type=int, default=seq_len, help='input sequence length of model encoder, look back window')
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
    parser.add_argument('--lradj', type=int, default=1,help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--save', type=bool, default =False, help='save the output results')
    parser.add_argument('--model_name', type=str, default=f'{model}')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--evaluate', type=bool, default=test_mode)  # only when you finished trainig

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
    print(args)

    torch.manual_seed(4321)  # reproducible
    torch.cuda.manual_seed_all(4321)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
    torch.backends.cudnn.enabled = True

    setting = '{}_{}_sl{}_pl{}_lr{}_bs{}_hid{}_s{}_l{}_dp{}_inv{}'.format(
            args.model,args.data, args.seq_len, args.pred_len,args.lr,
            args.batch_size,args.hidden_size,args.stacks, args.levels,args.dropout,args.inverse)

    exp = Experiment(args)  # set experiment object

    if not args.evaluate:
        print('Start training {}'.format(setting))
        exp.train(setting)

    print('Start Testing {}'.format(setting))
    acc, f1, acc_1h, f1_1h = exp.test(setting)    


    print('*'*41)
    print(f'Final Performance :\n'\
        #   f'acc:{acc}, f1:{f1},\n'\
          f'acc_1h:{acc_1h}, f1_1h:{f1_1h}')
    print('*'*41)
