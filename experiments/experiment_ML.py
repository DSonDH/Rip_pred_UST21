import itertools
import numpy as np

from sklearn.metrics import f1_score
from xgboost import XGBClassifier, XGBRFClassifier


def Experiment_ML(dataset_train: object, dataset_val: object, dataset_test: object,
                  pred_len: int = None, n_worker: int = 20, mode: str = None,
                  args: object = None
                  ) -> tuple:

    assert mode in ['seq2seq',
                    'single'], "mode should be one of 'seq2seq' or 'single'"
    assert pred_len != None, 'pred_len argument should not be None'

    X_train = dataset_train.X  # N x (T_in x C)
    y_train = dataset_train.y  # N x T_out
    X_val = dataset_val.X
    y_val = dataset_val.y
    X_test = dataset_test.X
    y_test = dataset_test.y

    assert X_train.ndim == 2 and X_val.ndim == 2 and X_test.ndim == 2
    assert y_train.ndim == 2 and y_val.ndim == 2 and y_test.ndim == 2

    # single prediction mode vs seq2seq mode
    y_train_toi = y_train[..., pred_len - 1] if mode == 'single' else y_train
    y_val_toi = y_val[..., pred_len - 1] if mode == 'single' else y_val
    y_test_toi = y_test[..., pred_len - 1] if mode == 'single' else y_test

    # HyperParameter tuning option setting
    if args.model_name == 'XGB':
        tuning_dict = {
            'eta': np.arange(0.05, 0.3, step=0.05),
            'gamma': np.arange(0, 0.02, step=0.01),
            'lambda': np.arange(1, 5, step=2),
            'max_depth': np.arange(3, 50, step=2),
            'max_leaves': np.arange(1, 31, step=3),
        }
    else:  #Random Forest
        tuning_dict = {
            'max_depth': np.arange(3, 31, step=3),
            'max_leaves': np.arange(1, 31, step=3),
            'subsample': [0.8],
            'colsample_bynode': [0.8],
            'num_parallel_tree': np.arange(100, 701, step=200),
            'max_depth': np.arange(3, 30, step=2),
        }

    default_dict = {
        'scale_pos_weight': len(y_train_toi) / y_train_toi.sum(),
        'objective': 'binary:logistic' if mode == 'single' else 'reg:squarederror',
        'early_stopping_rounds': 15 if args.model_name == 'XGB' else None,
        'n_estimators': 1000,
        'gpu_id': args.devices,
        'save_best': True,
        'seed': 1,
        'tree_method': 'gpu_hist',
        'verbosity': 0
    }

    # start learning
    best_f1 = 0.0
    hyperparameters = list(itertools.product(*tuning_dict.values()))
    for i, hp in enumerate(hyperparameters):
        print(f'--- {args.model_name}: HPO {i / len(hyperparameters) * 100:.1f}%'\
              ' is on processing ---')

        tuning_dict_tmp = {k: v for k, v in zip(tuning_dict.keys(), hp)}
        tuning_dict_tmp.update(default_dict)

        if args.model_name == 'XGB':
            classifier = XGBClassifier(**tuning_dict_tmp)
        else:
            classifier = XGBRFClassifier(**tuning_dict_tmp)

        classifier.fit(X_train,
                       y_train_toi,
                       eval_set=[(X_val, y_val_toi)]
                       )

        pred_val = classifier.predict(X_val)

        f1 = f1_score(y_val_toi.flatten(), pred_val.flatten(), zero_division=1.)
        if f1 >= best_f1:  # higher the f1, the better model
            best_f1 = f1
            best_hp = hp
            best_classifier = classifier

    pred_test = best_classifier.predict(X_test)

    return y_test_toi, np.array(pred_test)
