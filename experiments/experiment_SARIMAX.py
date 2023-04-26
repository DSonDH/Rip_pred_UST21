from typing import Tuple
import numpy as np
import statsmodels.api as sm
import itertools

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

from tqdm.contrib.concurrent import process_map
from functools import partial


def SARIMAX_multiprocess(i: int, 
                         pred_len: int=None, 
                         X_test: np.ndarray=None, 
                         y_test: np.ndarray=None,
                        ) -> np.ndarray:
    """tuning by changing model hyper parameter"""

    y_train_tmp = X_test[i, :, 10]
    x_train_tmp = X_test[i, :, :10]
    x_test_tmp = y_test[i, :, :10]
    
    # Fit the model
    p = range(0, 3)
    d = range(0, 3)
    q = range(0, 3)
    method_list = ['nm', 'lbfgs', 'powell']

    pdqs = list(itertools.product(p, d, q, method_list))
    best_aic = np.inf

    for p, d, q, method in pdqs:
        model = sm.tsa.statespace.SARIMAX(y_train_tmp,  # Time x 1
                                        exog=x_train_tmp,  # Time x m
                                        order=(p, d, q),
                                        )
        fit_res = model.fit(disp=False, 
                            maxiter=200,
                            method=method
                )
        # print(fit_res.mle_retvals)

        if fit_res.aic < best_aic:
            best_aic = fit_res.aic
            best_pdq = (p,  d, q)
            best_model = model
            best_fit_res = fit_res
    
    # print(best_fit_res.summary())
    # print(best_fit_res.mle_retvals)

    pred_test_regressor = best_fit_res.forecast(steps=pred_len, exog=x_test_tmp)
    # fcast_res1 = best_fit_res.get_forecast(steps=pred_len, exog=x_test)
    # fcast_res1.summary_frame()['mean']
    return np.clip(pred_test_regressor, 0, 1)


def Experiment_SARIMAX(dataset: object, pred_len: int=None, n_worker:int=20
                       )->Tuple:
    """
    fit test set data using SARIMAX algorithm and 
    calculate accuracyy, f1 score for all test instances.

    Args: 
        dataset: dataset object which have train, val, test datset with scaler
    Return:
        (testset prediction output, testset prediction label)
    """
    assert pred_len != None, 'pred_len argument should not be None'

    # Dataset

    #FIXME: sample 숫자 일부러 줄인거 없애기
    #FIXME: sample 숫자 일부러 줄인거 없애기
    X_test = dataset.X[:30, ...]  # N x 32 x 16
    y_test = dataset.y[:30, ...]  # N x 16 x 16
    y_test_label = y_test[:, :, 10]
    # print(X_test.shape,y_test.shape)

    assert(X_test.shape[2] >= 11)
    assert(y_test.shape[2] >= 11)

    # sample별 병렬로 결과 냄. tuning을 병렬화 하는게 아님.
    partial_wrapper = partial(SARIMAX_multiprocess, X_test=X_test, y_test=y_test, 
                                                    pred_len=pred_len)
    pred_test = process_map(partial_wrapper, range(len(X_test)),
                            max_workers=n_worker, 
                            chunksize=1)
    # 병렬처리 후 : 20 test sample의 SARIMAX 결과 내는데 1분 소요됨
    # list appended list (N x pred_len)
    return y_test_label, np.array(pred_test)


if __name__ == '__main__':
    Experiment_SARIMAX()

    '''
    order : iterable or iterable of iterables, optional
        The (p,d,q) order of the model for the number of AR parameters,
        differences, and MA parameters. `d` must be an integer
        indicating the integration order of the process, while
        `p` and `q` may either be an integers indicating the AR and MA
        orders (so that all lags up to those orders are included) or else
        iterables giving specific AR and / or MA lags to include. Default is
        an AR(1) model: (1,0,0).
    seasonal_order : iterable, optional
        The (P,D,Q,s) order of the seasonal component of the model for the
        AR parameters, differences, MA parameters, and periodicity.
        `D` must be an integer indicating the integration order of the process,
        while `P` and `Q` may either be an integers indicating the AR and MA
        orders (so that all lags up to those orders are included) or else
        iterables giving speci1fic AR and / or MA lags to include. `s` is an
        integer giving the periodicity (number of periods in season), often it
        is 4 for quarterly data or 12 for monthly data. Default is no seasonal
        effect.
        이안류 개별 instance들은 주기성 없으니까 이 옵션은 꺼야함 !!!
    trend : str{'n','c','t','ct'} or iterable, optional
        Parameter controlling the deterministic trend polynomial :math:`A(t)`.
        Can be specified as a string where 'c' indicates a constant (i.e. a
        degree zero component of the trend polynomial), 't' indicates a
        linear trend with time, and 'ct' is both. Can also be specified as an
        iterable defining the non-zero polynomial exponents to include, in
        increasing order. For example, `[1,1,0,1]` denotes
        :math:`a + bt + ct^3`. Default is to not include a trend component.
        이안류 개별 라벨들은 어떤 추세라고 할수있지? linear? constant? polynomial?
    '''