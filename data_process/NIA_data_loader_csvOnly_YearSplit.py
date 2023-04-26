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

        for x_start in range(self.input_len, len(df)):
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

        # FIXME: add onehot 효과 살펴보고, 효과 없으면 지우기
        # onehot = np.zeros(n_sites)
        # onehot[i] = 1
        # df[['site_' + item for item in site_names]] = onehot

        # train validation test split = 2 year: 1 year : 1 year
        Xy_train = df[: '2020-09-01 00:00']
        Xy_val = df['2020-09-01 00:00':'2021-09-01 00:00']
        Xy_test = df['2021-09-01 00:00':]

        # extract instances
        Xy_inst_tr, nonan_tr = self.get_instances(Xy_train)
        Xy_inst_val, nonan_val = self.get_instances(Xy_val)
        Xy_inst_te, nonan_te = self.get_instances(Xy_test)
        print(f'[{site}] tr:val:te = {len(Xy_inst_tr)} '
              f': {len(Xy_inst_val)} : {len(Xy_inst_te)}'
              )

        return Xy_inst_tr, Xy_inst_val, Xy_inst_val

    def __read_data__(self) -> None:
        """
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

        site_names = ['DC', 'HD', 'JM', 'NS', 'SJ']
        # incidence angle of each beach
        angle_inci_beach = [245, 178, 175, 47, 142]
        year = self.args.year

        if not os.path.exists(
            f'{self.root_path}/{self.NIA_work}_'
                f'processed_X_train_YearSplit.pkl'):
            print(f'no processed train file. start data preprocessing!! \n')

            csv_pth = f'{self.root_path}/obs_qc_100p'

            # Step2: loop each site to get instance for train, val, test years
            partial_wrapper = partial(self.get_instance_of_all_sites_multiprocess,
                                      csv_pth=csv_pth,
                                      angle_inci_beach=angle_inci_beach
                                      )
            meta = process_map(partial_wrapper,
                               range(len(site_names)),
                               site_names,
                               max_workers=5,
                               chunksize=1
                               )  # multiprocessing 하니깐 20배는 빨라짐
            meta = np.array(meta)

            assert meta.shape == (len(site_names), 3)  # site수 x tr/val/te 3종류

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
                f'{self.root_path}/{self.NIA_work}_NIA_train_scaler2D_YearSplit.pkl'
            )
            joblib.dump(
                self.scaler_3d,
                f'{self.root_path}/{self.NIA_work}_NIA_train_scaler3D_YearSplit.pkl'
            )
            
            joblib.dump(X_tr, f'{self.root_path}/{self.NIA_work}_processed_X_train_YearSplit.pkl')
            joblib.dump(y_tr, f'{self.root_path}/{self.NIA_work}_processed_y_train_YearSplit.pkl')
            joblib.dump(X_val, f'{self.root_path}/{self.NIA_work}_processed_X_val_YearSplit.pkl')
            joblib.dump(y_val, f'{self.root_path}/{self.NIA_work}_processed_y_val_YearSplit.pkl')
            joblib.dump(X_te, f'{self.root_path}/{self.NIA_work}_processed_X_test_YearSplit.pkl')
            joblib.dump(y_te, f'{self.root_path}/{self.NIA_work}_processed_y_test_YearSplit.pkl')
            # finished if block

        # when saved pre-processed file exist: just load files!
        self.X = joblib.load(
            f'{self.root_path}/{self.NIA_work}_processed_X_{self.flag}_YearSplit.pkl')
        self.y = joblib.load(
            f'{self.root_path}/{self.NIA_work}_processed_y_{self.flag}_YearSplit.pkl')

        if self.is_2d:
            self.scaler = joblib.load(  # train 기간에 대해 맞춰진 것
                f'{self.root_path}/{self.NIA_work}_NIA_train_scaler2D_YearSplit.pkl') 
            # (N, T, C) to (N, T * C)
            self.X = self.X.reshape(self.X.shape[0], -1)
            self.y = self.y.reshape(self.X.shape[0], -1)

        else:
            self.scaler = joblib.load(  # train 기간에 대해 맞춰진 것
                f'{self.root_path}/{self.NIA_work}_NIA_train_scaler3D_YearSplit.pkl')
        
        # apply normalization
        self.X = self.scaler.transform(self.X)
        self.y = self.scaler.transform(self.y)


    def __getitem__(self, index):
        return self.X[index, ...], self.y[index, ...]

    def __len__(self):
        return len(self.X)

    def inverse_transform(self, data, is_dnn=False):
        return self.scaler.inverse_transform(data, is_dnn)
