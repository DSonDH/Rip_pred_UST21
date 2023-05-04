from typing import Tuple
import numpy as np
import itertools
from sklearn import svm
from sklearn.metrics import f1_score
import time

def Experiment_SVM(dataset_train: object, dataset_val: object, dataset_test: object,
                   pred_len: int = None, toi: int = None, n_worker: int = 20
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
    assert toi != None, 'Time of Interst should not be None'

    X_train = dataset_train.X  # N x 32 x 16
    y_train = dataset_train.y  # N x 16 x 16
    X_val = dataset_val.X  # N x 32 x 16
    y_val = dataset_val.y  # N x 16 x 16
    X_test = dataset_test.X  # N x 32 x 16
    y_test = dataset_test.y  # N x 16 x 16

    assert X_train.ndim == 2 and X_val.ndim == 2 and X_test.ndim == 2
    assert y_train.ndim == 2 and y_val.ndim == 2 and y_test.ndim == 2

    y_train = dataset_train.scaler.inverse_transform(y_train)
    y_val = dataset_train.scaler.inverse_transform(y_val)
    y_test = dataset_train.scaler.inverse_transform(y_test)

    y_train_toi = y_train.reshape(-1, 11, pred_len)[..., 10, toi]
    y_val_toi = y_val.reshape(-1, 11, pred_len)[..., 10, toi]
    y_test_toi = y_test.reshape(-1, 11, pred_len)[..., 10, toi]
    
    #TODO: add more grid search parameters !!
    #     
    tuning_list = ['kernel', 'C', 'class_weight']
    kernel = ['linear', 'polynomial', 'rbf', 'sigmoid']  # 'linear' is not defined
    C = [0.1, 0.3, 0.5, 1, 2, 4, 10]
    class_weight = [None, 'balanced']
    # degree=3,
    # gamma="scale",
    # coef0=0.0,
    # shrinking=True,
    # probability=False,
    # tol=1e-3,
    # cache_size=200,
    # class_weight=None,
    # verbose=False,
    # max_iter=-1,
    # decision_function_shape="ovr",
    # break_ties=False,

    # start training, validation(HP tuning), and test
    hyperparameters = list(itertools.product(kernel, C, class_weight))
    
    # FIXME: hyperparamter 병렬화로 val f1, hp, classifier 반환 받아서 
    # best val_f1의 val_f1, hp, classifer만 추출한 뒤 그걸로 test 수행

    best_f1 = 0.0
    for hp in hyperparameters:  # loop 하나당 대충 10분 걸림
        classifier = svm.SVC()
        # svm.LinearSVC
        svm.NuSVC

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


if __name__ == '__main__':
    Experiment_SVM()
