import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import MinMaxScaler


def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean()

def Corr(pred, true):
    sig_p = np.std(pred, axis=0)
    sig_g = np.std(true, axis=0)
    m_p = pred.mean(0)
    m_g = true.mean(0)
    ind = (sig_g != 0)
    corr = ((pred - m_p) * (true - m_g)).mean(0) / (sig_p * sig_g)
    corr = (corr[ind]).mean()
    return corr

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def score_in_1h(pred, true, args): 
    '''
    input shape : N_iter x batchSize x 8(prediction length, 80min)
    즉 N_iter x batchsize 하면 total test instance 갯수가 됨.

    input : pred(scaled), true(scaled)
    output : probability of one batch
    '''

    true = true.reshape(-1, true.shape[-1])
    pred = pred.reshape(-1, pred.shape[-1])
    
    mmx_fn = MinMaxScaler()
    mmx_fn.fit(true)
    true_scale = mmx_fn.transform(true)
    pred_scale = mmx_fn.transform(pred)

    pred_bin = 1 * (pred_scale >= 0.5)  # y.astype(int) 

    true_flat = true_scale.flatten()
    pred_bin_flat = pred_bin.flatten()
    
    f1 = f1_score(true_flat, pred_bin_flat)
    acc = accuracy_score(true_flat, pred_bin_flat)
    cm = confusion_matrix(true_flat, pred_bin_flat) 
    # print(f'[all time point] \n {cm}, row : obs, columns : pred')

    # 마지막 time point에 대해서만 성능 평가하는 방식
    true_1h = true[:, -1]
    pred_1h = pred_bin[:, -1]
    acc_1h = accuracy_score(true_1h, pred_1h)
    f1_1h = f1_score(true_1h, pred_1h)
    cm = confusion_matrix(true_1h, pred_1h) 
    print(f'[1H time point] \n {cm}, row : obs, columns : pred')

    # 1h 내 이안류 발생이 한번이라도 있으면 일괄 적용 방식
    # pts_in_1h = 60//args.itv
    # pred_max = np.max(pred_bin[:,:pts_in_1h], axis=1)
    # true_max = np.max(true2[:,:pts_in_1h], axis=1)
    # acc_1h = np.sum(pred_max == true_max)/pred_max.size

    # pred_max_flat = pred_max.flatten()
    # true_max_flat = true_max.flatten()
    # cm_max = confusion_matrix(true_max_flat, pred_max_flat) 
    # print(f'{cm_max}, row : obs, pred : columns')
    # f1_1h = f1_score(true_max_flat, pred_max_flat)

    return acc, f1, acc_1h, f1_1h, true, pred, true_1h, pred_1h

    """
    # old way
    pred_round = pred.round()

    # get initial 6*10 minutes and get max value
    pred_max = np.max(pred_round[:,:pts_in_1h], axis=1)
    true_max = np.max(true[:,:pts_in_1h], axis=1)
    assert np.max(pred_max, axis=None) in [0, 1]

    acc_1h = np.sum(pred_max == true_max)/pred_max.size
    # accuracy_score(true_flat, pred_flat)

    pred_round_flat = pred_round.flatten()
    true_flat = true.flatten()
    pred_max_flat = pred_max.flatten()
    true_max_flat = true_max.flatten()
    cm_max = confusion_matrix(true_max_flat, pred_max_flat) 
    print(f'{cm_max}, row : obs, pred : columns')

    f1_1h = f1_score(true_max_flat, pred_max_flat)
    
    f1 = f1_score(pred_round_flat, true_flat)
    acc = accuracy_score(pred_round_flat, true_flat)
    confusion_matrix(true_flat, pred_round_flat) 
    
    return acc, f1, acc_1h, f1_1h
    """


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    #corr1 = CORR(pred, true)
    corr = Corr(pred, true)
    return mae,mse,rmse,mape,mspe,corr
