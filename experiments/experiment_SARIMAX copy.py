from typing import Tuple
import numpy as np
import statsmodels.api as sm
import itertools
from metrics.NIA_metrics import metric_classifier, metric_regressor

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

from tqdm.contrib.concurrent import process_map

def Experiment_SARIMAX(dataset: object)->Tuple:
    """
    fit test set data using SARIMAX algorithm and 
    calculate accuracyy, f1 score for all test instances.

    Args: 
        dataset: dataset object which have train, val, test datset with scaler
    """
    # Dataset
    X_test = dataset.X_test  # N x 32 x 16
    y_test = dataset.y_test  # N x 16 x 16
    # print(X_test.shape,y_test.shape)

    assert(X_test.shape[2] >= 11)
    assert(y_test.shape[2] >= 11)

    pred_test = []
    for i in range(len(X_test)):

        # if i * 1000 == 0:
        #     print(f'SARIMAX evaluation : {i / len(X_test) * 100: .2f}% done ')

        y_train_tmp = X_test[i, :, 11]
        x_train_tmp = X_test[i, :, :10]
        x_test_tmp = y_test[i, :, :10]
        y_test_label = y_test[i, :, 11]
        pred_len = dataset.pred_len

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

            # method : str, optional
            #     The `method` determines which solver from `scipy.optimize`
            #     is used, and it can be chosen from among the following strings:

            #     - 'newton' for Newton-Raphson
            #     - 'nm' for Nelder-Mead
            #     - 'bfgs' for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
            #     - 'lbfgs' for limited-memory BFGS with optional box constraints
            #     - 'powell' for modified Powell's method
            #     - 'cg' for conjugate gradient
            #     - 'ncg' for Newton-conjugate gradient
            #     - 'basinhopping' for global basin-hopping solver

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
        
        pred_test.extend(
            np.clip(pred_test_regressor, 0, 1)
        )
    
    acc, f1 = metric_classifier(np.array(y_test_label), np.array(pred_test))
    # dummy = metric_regressor(np.array(y_test), pred_test.values)

    return acc, f1


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