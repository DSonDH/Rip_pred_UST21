from functools import partial
from tqdm.contrib.concurrent import process_map
from typing import Tuple
import numpy as np
from sklearn import svm
import itertools

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)


def Experiment_SVM(dataset: object, pred_len: int = None, n_worker: int = 20
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

    # Dataset

    # FIXME: [:30, ...]
    X_test = dataset.X  # N x 32 x 16
    y_test = dataset.y  # N x 16 x 16
    
    assert X_test.shape[2] == 11 and y_test.shape[2] == 11

    clf = svm.SVC()
    clf.fit()
    clf.predict(X_test)
    
    return y_test, np.array(pred_test)


if __name__ == '__main__':
    Experiment_SVM()
