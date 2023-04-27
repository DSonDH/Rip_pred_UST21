import numpy as np
import pandas as pd
import joblib

from itertools import product
from sklearn.feature_selection import mutual_info_classif, r_regression


data_pth = 'datasets/NIA/ripcurrent_100p_processed_X_train_allYear_YearSplit.pkl'
label_pth = 'datasets/NIA/ripcurrent_100p_processed_y_train_allYear_YearSplit.pkl'
pred_idx = 6

scaler_pth = 'datasets/NIA/ripcurrent_100p_NIA_train_AllPorts_allYear_scaler_YearSplit.pkl'
scaler = joblib.load(scaler_pth)

data = joblib.load(f'{data_pth}')
data = scaler.inverse_transform(data)[:, ..., :11]
label = joblib.load(f'{label_pth}')
label = label[..., pred_idx, 10]

rows = []
for i in range(11):
    data_tmp = data[:, ..., i]

    for importance_type in ['mic','r_reg']:
        value = {'mic': mutual_info_classif,
                'r_reg': r_regression}[importance_type](data_tmp, label.flatten())[0]
        row = {'index':i, 'importance_type':importance_type, 'value':value}
        rows.append(row)

table = pd.DataFrame(rows)
# id_vars = ['station_name', 'pred_hour', 'importance_type', 'variable']
# table = table.set_index(id_vars).unstack().droplevel(0,axis=1).reset_index()
# idx = np.arange(0, len(table), 2)
# table = table.iloc[idx]
table.to_csv('./feature_importance_before_training.csv', index=False)
