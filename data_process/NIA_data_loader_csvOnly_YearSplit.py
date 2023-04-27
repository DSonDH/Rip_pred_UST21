from typing import List, Set, Dict, Tuple, Union
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from utils.tools import StandardScaler

from tqdm.contrib.concurrent import process_map
from functools import partial

warnings.filterwarnings('ignore')


class Dataset_NIA(Dataset):
    """ 
    if saved meta data is not exist,
    load dataframe and preprocess instances
    save 2d, 3d scaler and train / validation / test instances

    # implementation details 
    csv : 매년 6월01일 00시 ~ 8월 31일 23시 55분까지 5분간격.

    train/val/test split 방식:
        Split Way1: tr, val, test 기간 따로 정하여 나누는 방법.  <--- Current Way
        Split Way2: 8:1:1 timestamp 갯수로 나누는 방법.  <--- Improper Way

    mode통합 instance 추출:
        train set, val set, test set 힌번에 추출하고 저장하는 방식
        즉 training phase에 val, test set 추출해서 
        val, test시에는 저장된거 불러오기만 함

    instance 추출:
        all_site 분리해서 샘플 추출함 (항별로 연속된 시간이 아니므로)

    In case of training dataset extaction:
        Read 'DC' --> train val test
                        +
        Read 'HD' --> train val test
                        +
        Read 'JM' --> train val test
                        +
        Read 'NS' --> train val test
                        +
        Read 'SJ' --> train val test
                        ||
        merge final : train val test

        fit scaler using merged train set
        save the scaler object as pickle
        when loading saved file : the file is NOT normalized
        You should normalize first when loading saved instances!    
    """
    def __init__(self, args: dict=None, flag: str=None, is_2d: bool=False,
                 ) -> None:
        assert flag != None, "Please specify 'train' or 'val' or 'test' flag"
        assert args.pred_len != None and args.input_len != None, \
            'Please specify input_len and pred_len'

        # init
        if args.model_name == 'SCINet':
            # !!!! 모델에 2^n 제곱 길이만 들어갈 수 있으므로 10분간격 시간단위(6의배수)는 불가능
            self.input_len = 2 * 8 * 2  # lagging len
            self.pred_len = 8 * 2
        else:
            self.input_len = args.input_len
            self.pred_len = args.pred_len
        self.input_dim = args.input_dim
        self.NIA_work = args.NIA_work
        self.root_path = args.root_path
        self.data_path = args.data_path
        self.data = args.data
        self.port = args.port
        self.args = args
        self.flag = flag
        self.is_2d = is_2d

        self.__read_data__()


    def load_df(self,
                csv_pth: str,
                site: str,
                angle_inci_beach: int,
                input_dim: int
                ) -> pd.DataFrame:

        df = pd.read_csv(f'{csv_pth}/{site}-total.csv', encoding='euc-kr')
        df = df.iloc[:, 1:]
        df['ob time'] = pd.to_datetime(df['ob time'])
        df = df.set_index('ob time')
        if input_dim < 11:
            # TODO: find no specified features and selctively drop them
            df = df.drop(columns=df.columns[5])
        df = df.fillna(method='ffill', limit=2).fillna(method='bfill', limit=2)
        df['wave direction'] -= angle_inci_beach  # wave direction 보정
        return df


    def get_instances(self, df: pd.DataFrame) -> Tuple[List, int]:
        instance_list = []
        nonan = 0
        assert len(df) >= (self.input_len + self.pred_len), \
            'not enough time to extract instance !'

        for x_start in range(0, len(df) - self.input_len - self.pred_len):
            y_end = x_start + self.input_len + self.pred_len

            # 일단 X, y합쳐서 뽑고, 나중에 분리
            Xy_instance = df.iloc[x_start:y_end, :]
            if not Xy_instance.isnull().values.any():
                instance_list.append(Xy_instance.values)
            else:
                nonan += 1

        return np.array(instance_list), nonan


    def get_instance_of_all_sites_multiprocess(self,
                                               i: int,
                                               site: str,
                                               csv_pth: str,
                                               angle_inci_beach: List
                                               ) -> Tuple:
        """ get all instance of each site """
        print(f' ## Start instance extraction process: [{site}]')

        df = self.load_df(csv_pth, site, angle_inci_beach[i], self.input_dim)

        # adding onehot was not helpful.
        # onehot = np.zeros(n_sites)
        # onehot[i] = 1
        # df[['site_' + item for item in site_names]] = onehot

        # train validation test split = 2 year: 1 year : 1 year
        Xy_train = df[ : '2020-09-01 00:00']
        Xy_val = df['2020-09-01 00:00' : '2021-09-01 00:00']
        Xy_test = df['2021-09-01 00:00' : ]

        # extract instances
        Xy_inst_tr, nonan_tr = self.get_instances(Xy_train)
        Xy_inst_val, nonan_val = self.get_instances(Xy_val)
        Xy_inst_te, nonan_te = self.get_instances(Xy_test)
        print(f'[{site}] tr:val:te = {len(Xy_inst_tr)} '
              f': {len(Xy_inst_val)} : {len(Xy_inst_te)}'
              )

        return Xy_inst_tr, Xy_inst_val, Xy_inst_te


    def __read_data__(self) -> None:
        site_names = ['DC', 'HD', 'JM', 'NS', 'SJ']
        angle_inci_beach = [245, 178, 175, 47, 142]  # incidence angle of each beach

        if not os.path.exists(
            f'{self.root_path}/{self.NIA_work}_'
                f'meta_X_train_{self.input_len}_{self.pred_len}.pkl'):

            print(f'no processed train file. start data preprocessing!! \n')

            csv_pth = f'{self.root_path}/obs_qc_100p'

            # Step2: loop each site to get instance for train, val, test years
            partial_wrapper = partial(self.get_instance_of_all_sites_multiprocess,
                                      csv_pth=csv_pth,
                                      angle_inci_beach=angle_inci_beach
                                      )
            # TODO: process_map seems to cause "OSError: handle is closed"
            # when main.py procedure ends. No problem is caused while 
            # main.py is running.
            meta = process_map(partial_wrapper,
                               range(len(site_names)),
                               site_names,
                               max_workers=5,
                               chunksize=1
                               )  # multiprocessing leads to about 20X faster processing
            meta = np.array(meta)

            assert meta.shape == (len(site_names), 3)  # num_site x 3(tr/val/te)

            # merge each train set, validation set, test set
            Xy_train = np.concatenate(meta[:, 0], axis=0)
            Xy_val = np.concatenate(meta[:, 1], axis=0)
            Xy_test = np.concatenate(meta[:, 2], axis=0)

            # Step3: split X and y
            X_tr = Xy_train[:, :self.input_len, :]
            y_tr = Xy_train[:, self.input_len:, :]
            X_val = Xy_val[:, :self.input_len, :]
            y_val = Xy_val[:, self.input_len:, :]
            X_te = Xy_test[:, :self.input_len, :]
            y_te = Xy_test[:, self.input_len:, :]

            self.scaler_2d = StandardScaler(is_2d=True)
            self.scaler_3d = StandardScaler(is_2d=False)
            self.scaler_2d.fit(X_tr)
            self.scaler_3d.fit(X_tr)

            joblib.dump(
                self.scaler_2d,
                f'{self.root_path}/{self.NIA_work}_scaler2D_{self.input_len}_{self.pred_len}.pkl'
            )
            joblib.dump(
                self.scaler_3d,
                f'{self.root_path}/{self.NIA_work}_scaler3D_{self.input_len}_{self.pred_len}.pkl'
            )

            joblib.dump(X_tr,
                        f'{self.root_path}/{self.NIA_work}_meta_X_train_{self.input_len}_{self.pred_len}.pkl')
            joblib.dump(y_tr,
                        f'{self.root_path}/{self.NIA_work}_meta_y_train_{self.input_len}_{self.pred_len}.pkl')
            joblib.dump(X_val,
                        f'{self.root_path}/{self.NIA_work}_meta_X_val_{self.input_len}_{self.pred_len}.pkl')
            joblib.dump(y_val,
                        f'{self.root_path}/{self.NIA_work}_meta_y_val_{self.input_len}_{self.pred_len}.pkl')
            joblib.dump(X_te,
                        f'{self.root_path}/{self.NIA_work}_meta_X_test_{self.input_len}_{self.pred_len}.pkl')
            joblib.dump(y_te,
                        f'{self.root_path}/{self.NIA_work}_meta_y_test_{self.input_len}_{self.pred_len}.pkl')
            
            # record N_Xy, class ratio as input_len and pred_len changes
            df = pd.DataFrame()
            df.loc[0, 0] = len(X_tr)
            df.loc[1, 0] = y_tr[..., 10].sum() / y_tr.size
            df.loc[2, 0] = len(X_val)
            df.loc[3, 0] = y_val[..., 10].sum() / y_val.size
            df.loc[4, 0] = len(X_te)
            df.loc[5, 0] = y_te[..., 10].sum() / y_te.size
            df.to_csv(f'./results/NSample_analysis_{self.NIA_work}_{self.input_len}_{self.pred_len}.csv')
            # finished if block

        # when saved pre-processed file exist: just load files!
        self.X = joblib.load(
            f'{self.root_path}/{self.NIA_work}_meta_X_{self.flag}_{self.input_len}_{self.pred_len}.pkl')
        self.y = joblib.load(
            f'{self.root_path}/{self.NIA_work}_meta_y_{self.flag}_{self.input_len}_{self.pred_len}.pkl')

        if self.is_2d:
            self.scaler = joblib.load(
                f'{self.root_path}/{self.NIA_work}_scaler2D_{self.input_len}_{self.pred_len}.pkl')
            self.X = self.X.reshape(self.X.shape[0], -1)  # (N, T, C) to (N, T * C)
            self.y = self.y.reshape(self.X.shape[0], -1)  # (N, T, C) to (N, T * C)
        else:
            self.scaler = joblib.load(
                f'{self.root_path}/{self.NIA_work}_scaler3D_{self.input_len}_{self.pred_len}.pkl')

        self.X = self.scaler.transform(self.X)
        self.y = self.scaler.transform(self.y)


    def __getitem__(self, index):
        return self.X[index, ...], self.y[index, ...]


    def __len__(self):
        return len(self.X)


    def inverse_transform(self, data, is_dnn=False):
        return self.scaler.inverse_transform(data, is_dnn)
