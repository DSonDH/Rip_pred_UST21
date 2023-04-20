from typing import List, Union, Tuple, Dict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def metric_classifier(true: np.ndarray, pred: np.ndarray) -> Dict: 
    """
    true, pred 를 flatten하고 binarize 하여 2분류 판별 성능 계산

    Args:
        true : np.ndarray
            N x 1 또는 N x 3 또는 N x pred_len 모양일 것임
            shape이 무엇이던 flatten 하므로 상관없음.
        pred : np.ndarray
            shape이 무엇이던 flatten 하므로 상관없음.
    Return:
        A dictionary containing; acc, f1,TP, FP, FN, TN
    """
    assert len(true) > 0 and len(pred) > 0, \
        'Error of metric_all(): true or pred length is zero'
    
    if type(true) != np.ndarray:
        true = np.array(true)
    if type(pred) != np.ndarray:
        pred = np.array(pred)

    true = true.flatten()
    pred = pred.flatten()

    # convert probability to label
    pred_bin = 1 * (pred >= 0.5)  # y.astype(int) 
    
    # TP, FP, FN, TN order
    cm = confusion_matrix(true, pred_bin).flatten()
    
    if len(cm) == 1:  # make dummy zeros
        if true[0] == 1:
            cm = np.array([len(true), 0, 0, 0])
        else:
            cm = np.array([0, 0, 0, len(true)])

    result_dict = {}
    result_dict['acc'] = accuracy_score(true, pred_bin)
    result_dict['f1'] = f1_score(true, pred_bin, zero_division=1.)
    result_dict['TP'] = cm[0]
    result_dict['FP'] = cm[1]
    result_dict['FN'] = cm[2]
    result_dict['TN'] = cm[3]

    return result_dict
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


def Corr(true: np.ndarray, pred: np.ndarray) -> float:
    sig_p = np.std(pred, axis=0)
    sig_g = np.std(true, axis=0)
    m_p = pred.mean(0)
    m_g = true.mean(0)
    ind = (sig_g != 0)
    corr = ((pred - m_p) * (true - m_g)).mean(0) / (sig_p * sig_g)
    corr = (corr[ind]).mean()
    return corr

def metric_regressor(true: np.ndarray, pred: np.ndarray) -> Dict:
    """ true, pred로 성능 계산
    Args:
        true : np.ndarray
            N x 1 또는 N x 3 또는 N x pred_len 모양일 것임
            shape이 무엇이던 상관없음.
        pred : np.ndarray
            shape이 무엇이던 상관없음.
    
    Return:
        A dictionary containing; acc, f1,TP, FP, FN, TN

    ❗❗❕❕❕❗❗깨달음 노트
        np.mean하면 N, T함께 계산(flatten해서 계산) T축 먼저 metric 계산해서 
        instance별 성능 뽑고 이 성능을 평균내는거 아닌가 싶었는데, 
        수식으로 써보면 같음 ㄷㄷㄷ
        즉, time series를 따로 뗴서 하나의 instance로 봐도 무방하다는 것임.
        즉, 시계열 추세에 맞지않는 metric이고, 이를 개선하면 논문감? ㅎㅎ
    """
    assert len(true) > 0 and len(pred) > 0, \
        'Error of metric_all(): true or pred length is zero'
    
    if type(true) != np.ndarray:
        true = np.array(true)
    if type(pred) != np.ndarray:
        pred = np.array(pred)
    
    result_dict = {}
    result_dict['mae'] = np.mean(np.abs(pred - true))
    result_dict['mse'] = np.mean((pred - true) ** 2)
    result_dict['rmse'] = np.sqrt(result_dict['mse'])
    result_dict['mape'] = np.mean(np.abs((pred - true) / true))
    result_dict['mspe'] = np.mean(np.square((pred - true) / true))
    result_dict['corr'] = Corr(true, pred)

    return result_dict


def metric_all(true: np.ndarray, pred: np.ndarray) -> Dict: 
    """ 
    from metric_classifier() and metric_regressor(),
    get all metric as dictionary items and return a metrics
    """
    assert len(true) > 0 and len(pred) > 0, \
        'Error of metric_all(): true or pred length is zero'

    if type(true) != np.ndarray:
        true = np.array(true)
    if type(pred) != np.ndarray:
        pred = np.array(pred)
    
    metrics = metric_classifier(true, pred)
    metrics.update(metric_regressor(true, pred))
    
    return metrics
