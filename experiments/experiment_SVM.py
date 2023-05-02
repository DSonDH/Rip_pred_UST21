from typing import Tuple
import numpy as np
import itertools
from sklearn import svm
from sklearn.metrics import f1_score


def Experiment_SVM(dataset_train: object, dataset_val: object, dataset_test: object,
                   pred_len: int = None, n_worker: int = 20
                   ) -> Tuple:
    """
    fit test set data using SVM algorithm and 
    calculate accuracyy, f1 score for all test instances.

    Args: 
        dataset: dataset object which have train, val, test datset with scaler
    Return:
        (testset prediction output, testset prediction label)
    """
    assert pred_len != None, 'pred_len argument should not be None'

    X_train = dataset_train.X  # N x 32 x 16
    y_train = dataset_train.y  # N x 16 x 16
    X_val = dataset_val.X  # N x 32 x 16
    y_val = dataset_val.y  # N x 16 x 16
    X_test = dataset_test.X  # N x 32 x 16
    y_test = dataset_test.y  # N x 16 x 16

    assert X_train.ndim == 2 and X_val.ndim == 2 and X_test.ndim == 2
    assert y_train.ndim == 2 and y_val.ndim == 2 and y_test.ndim == 2

    # Support Vector Machine algorithms are not scale invariant,
    # so it is highly recommended to scale your data.
    # my dataloader first normalize them first.
    
    tuning_list = ['kernel', 'C', 'class_weight']
    kernel = ['linear', 'polynomial', 'rbf', 'sigmoid']  # 'linear' is not defined
    C = [0.5, 1, 2]
    class_weight = [None, 'balanced']
    # C=1.0,
    # kernel="rbf",
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
    # random_state=None,

    # start training, validation(HP tuning), and test
    hyperparameters = list(itertools.product(kernel, C, class_weight))
    best_hp = None
    best_f1 = 0.0
    for hp in hyperparameters:
        classifier = svm.SVC()
        # svm.LinearSVC
        # svm.LinearSVR
        # svm.NuSVC
        # svm.NuSVR

        for k, v in zip(tuning_list, hp):
            eval(f"classifier.set_params({k}={v})")

        classifier.fit(X_train, y_train)

        pred_val = classifier.predict(X_val)
        
        f1 = f1_score(y_val, pred_val)

        # higher the f1, the better model
        if f1 > best_f1:
            best_f1 = f1
            best_hp = hp
            best_classifier = classifier


    pred_test = best_classifier.predict(X_test)

    return y_test, np.array(pred_test)


if __name__ == '__main__':
    Experiment_SVM()
