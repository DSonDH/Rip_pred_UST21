from typing import Tuple
import numpy as np
import itertools
from sklearn import svm
from sklearn.metrics import f1_score

from functools import partial
from tqdm.contrib.concurrent import process_map
import time


def SVM_multiprocess(hp: tuple,
                         tuning_list: list = None,
                         X_train: np.ndarray = None,
                         y_train_toi: np.ndarray = None,
                         X_val: np.ndarray = None,
                         y_val_toi: np.ndarray = None,
                         ) -> list:
    """tuning by changing model hyper parameter"""
    classifier = svm.SVC()

    for k, v in zip(tuning_list, hp):
        if v == None:
            continue
        if isinstance(v, str):
            eval(f"classifier.set_params({k}='{v}')")
        else:  # numeric types
            eval(f"classifier.set_params({k}={v})")

    classifier.fit(X_train, y_train_toi)  # 7min 30sec for one fitting.
    
    pred_val = classifier.predict(X_val)

    f1 = f1_score(y_val_toi, pred_val)

    return f1, hp, classifier


def Experiment_SVM(dataset_train: object, dataset_val: object, dataset_test: object,
                   pred_len: int = None, n_worker: int = 20
                   ) -> Tuple:
    """
    fit test set data using SVM algorithm and 
    calculate accuracyy, f1 score for all test instances.

    Args: 
        dataset: dataset object which have train, val, test datset with scaler
        toi (int) : time of interest. time to implement prediciton.
    Return:
        (testset prediction output, testset prediction label)
    """
    assert pred_len != None, 'pred_len argument should not be None'
    
    X_train = dataset_train.X # N x (T_in x C)
    y_train = dataset_train.y  # N x (T_out x C)
    X_val = dataset_val.X
    y_val = dataset_val.y
    X_test = dataset_test.X
    y_test = dataset_test.y

    assert X_train.ndim == 2 and X_val.ndim == 2 and X_test.ndim == 2
    assert y_train.ndim == 2 and y_val.ndim == 2 and y_test.ndim == 2
    
    y_train_toi = y_train[..., pred_len - 1]
    y_val_toi = y_val[..., pred_len - 1]
    y_test_toi = y_test[..., pred_len - 1]

    # 320 combinataions => 320 * 7.5 hours = 100 days! *친
    tuning_list = ['kernel', 'C', 'class_weight']
    kernel = ['linear', 'poly', 'rbf']
    C = [0.1, 0.5, 1, 5]
    class_weight = [None, 'balanced']
    # {‘scale’, ‘auto’} or float  # only for ‘rbf’, ‘poly’ and ‘sigmoid’.

    # degree=3  # only for kerner = 'poly'
    # coef0=0.0  # ndependent term in kernel function. only for ‘poly’ and ‘sigmoid’.

    # tol=1e-3,
    # max_iter=-1,

    # shrinking=True,
    # probability=False,
    # cache_size=200,

    # start training, validation(HP tuning), and test
    hyperparameters = list(itertools.product(kernel, C, class_weight))

    partial_wrapper = partial(SVM_multiprocess,
                              tuning_list=tuning_list,
                              X_train=X_train,
                              y_train_toi=y_train_toi,
                              X_val=X_val,
                              y_val_toi=y_val_toi,
                              )
    tuning_results = process_map(partial_wrapper,
                                 hyperparameters,
                                 max_workers=n_worker,
                                 chunksize=1
                                 )
    
    best_idx = np.argmax([item[0] for item in tuning_results])
    best_f1_val = tuning_results[best_idx][0]
    best_hp = tuning_results[best_idx][1]
    best_classifier = tuning_results[best_idx][2]

    pred_test = best_classifier.predict(X_test)

    return y_test_toi, np.array(pred_test)
    
    """ no multiprocessing mode
    best_f1 = 0.0
    for hp in hyperparameters:  # loop 하나당 대충 10분 ~ 7시간30분 걸림
        classifier = svm.SVC()
        # svm.LinearSVC
        # svm.NuSVC

        for k, v in zip(tuning_list, hp):
            if v == None:
                continue
            if isinstance(v, str):
                eval(f"classifier.set_params({k}='{v}')")
            else: # numeric types
                eval(f"classifier.set_params({k}={v})")

        tic = time.time()
        classifier.fit(X_train, y_train_toi)  # 7min 30sec for one fitting.
        toc = time.time()
        print(f'time for one fitting : {(toc - tic) // 60}min {(toc - tic) % 60}sec')

        pred_val = classifier.predict(X_val)

        f1 = f1_score(y_val_toi, pred_val)
        if f1 > best_f1:  # higher the f1, the better model
            best_f1 = f1
            best_hp = hp
            best_classifier = classifier  # 48byte
    pred_test = best_classifier.predict(X_test)

    return y_test_toi, np.array(pred_test)
    """


if __name__ == '__main__':
    Experiment_SVM()
