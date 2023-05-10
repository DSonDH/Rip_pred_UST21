import itertools
import numpy as np

from sklearn.metrics import f1_score
from xgboost import XGBClassifier, XGBRFClassifier


def Experiment_ML(dataset_train: object, dataset_val: object, dataset_test: object,
                  pred_len: int = None, n_worker: int = 20, mode: str = None
                  ) -> tuple:
    assert mode in ['seq2seq',
                    'single'], "mode should be one of 'seq2seq' or 'single'"
    assert pred_len != None, 'pred_len argument should not be None'

    X_train = dataset_train.X  # N x (T_in x C)
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

    # TODO: HPO range 설정, gpu1번 쓰도록 설정, xgb/rf 선택하도록 if문 설정
    # FIXME: multi-output option 설정
    sdfsdfsdfdsf
    
    # sample_weight: Optional[ArrayLike] = None,
    # eval_metric: Optional[Union[str, Sequence[str], Metric]] = None,
    tuning_dict = {
        'eta': np.arange(0.05, 0.3, step=0.05),
        'gamma': np.arange(0, 0.02, step=0.01),
        'lambda': np.arange(1, 5, step=2),
        'max_depth': np.arange(3, 50, step=2),
        'max_leaves': np.arange(1, 31, step=3),
    }
    default_dict = {
        'early_stopping_rounds': 15,
        'n_estimators': 1000,
        'gpu_id': 1,
        'save_best': True,
        'seed': 1,
        'tree_method': 'gpu_hist',
        'verbosity': 0
    }

    best_f1 = 0.0
    hyperparameters = list(itertools.product(*tuning_dict.values()))
    for hp in hyperparameters:

        tuning_dict_tmp = {k: v for k, v in zip(tuning_dict.keys(), hp)}
        tuning_dict_tmp.update(default_dict)

        classifier = XGBClassifier(**tuning_dict_tmp)
        classifier.fit(X_train,
                       y_train_toi,
                       eval_set=[(X_val, y_val_toi)]
                       )

        pred_val = classifier.predict(X_val)

        f1 = f1_score(y_val_toi, pred_val)
        if f1 > best_f1:  # higher the f1, the better model
            best_f1 = f1
            best_hp = hp
            best_classifier = classifier

    pred_test = best_classifier.predict(X_test)

    return y_test_toi, np.array(pred_test)
