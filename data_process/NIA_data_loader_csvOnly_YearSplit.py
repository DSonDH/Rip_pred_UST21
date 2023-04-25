from typing import List, Set, Dict, Tuple, Union
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from utils.tools import StandardScaler

# from tqdm.contrib.concurrent import process_map
# from multiprocessing import cpu_count

warnings.filterwarnings('ignore')


class Dataset_NIA(Dataset):
    def __init__(self, args: dict=None, flag: str=None) -> None:
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

        self.__read_data__()


    def load_df(self, 
                csv_pth: str, 
                site: str, 
                angle_inci_beach: int,
                input_dim: int
               ) -> pd.DataFrame:

        
        df = pd.read_csv(f'{csv_pth}/{site}-total.csv', encoding='euc-kr')
        df = df.iloc[:,1:]
        df['ob time'] = pd.to_datetime(df['ob time'])
        df = df.set_index('ob time')
        if input_dim < 11:
            #TODO: find no specified features and selctively drop them
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


    def __read_data__(self) -> None:
        """
        csv : 매년 6월01일 00시 ~ 8월 31일 23시 55분까지 5분간격.
        
        train/val/test split 방식:
            Split Way1: tr, val, test 기간 따로 정하여 나누는 방법.  <--- Current Way
            Split Way2: 8:1:1 timestamp 갯수로 나누는 방법.  <--- Improper Way

        mode통합 instance 추출:
            train, val, test 힌번에 구하고 저장하는 방식
        
        instance 추출:
            all_site 분리해서 샘플 추출함 (항별로 연속된 시간이 아니므로)

        In case of training dataset extaction:
            Read 'DC' --> train val test
                            +
            Read 'HD' --> train val test
                          self.  +
            Read 'JM' --> train val test
                            +
            Read 'NS' --> train val test
                            +
            Read 'SJ' --> train val test
                            ||
            merge final : train val test
            
            get normalizer using merged train set
            save the normalized files
            when loading saved file : the file is already normalized
        """
        self.scaler = StandardScaler() #MinMaxScaler()
        site_names = ['DC', 'HD', 'JM', 'NS', 'SJ']
        angle_inci_beach = [245, 178, 175, 47, 142]  # incidence angle of each beach
        year = self.args.year

        if not os.path.exists(
            f'{self.root_path}/{self.NIA_work}_'\
            f'processed_X_train_{year}_YearSplit.pkl'):
            print(f'no processed train file. start data preprocessing!! \n')
            
            csv_pth = f'{self.root_path}/obs_qc_100p'

            # Step2: loop each site to get instance and split tr/val/te = 8:1:1
            X_tr = []
            y_tr = []
            X_val = []
            y_val = []
            X_te = []
            y_te = []
            n_sites = len(site_names)
            for i, site in enumerate(site_names):
                print(f' ## processing {site} for train')

                df = self.load_df(csv_pth, site, angle_inci_beach[i], self.input_dim)
                
                # print label ratio
                n1 = sum(df['RipLabel'] == 1)
                n0 = sum(df['RipLabel'] == 0)
                print(f'N_raw instance = {n1 + n0}')
                print(f'N_0 instance = {n0}, N_1 instance = {n1}')
                print(f'N1 / N_0 ratio = {n1 / (n0 + n1) * 100:.2f} \n')

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
                print(f'[{site}] tr:val:te = {len(Xy_inst_tr)} '\
                      f': {len(Xy_inst_val)} : {len(Xy_inst_te)}'
                     )

                # detach X, y seperately and append all site together
                X_tr.append(Xy_inst_tr[:, :self.input_len, :])
                y_tr.append(Xy_inst_tr[:, self.input_len:, :])
                X_val.append(Xy_inst_val[:, :self.input_len, :])
                y_val.append(Xy_inst_val[:, self.input_len:, :])
                X_te.append(Xy_inst_te[:, :self.input_len, :])
                y_te.append(Xy_inst_te[:, self.input_len:, :])

            # Step3: merge array finally
            X_tr = np.concatenate(X_tr, axis=0)
            y_tr = np.concatenate(y_tr, axis=0)
            X_val = np.concatenate(X_val, axis=0)
            y_val = np.concatenate(y_val, axis=0)
            X_te = np.concatenate(X_te, axis=0)
            y_te = np.concatenate(y_te, axis=0)

            # Step1: merge train set each site and get scaler first
            self.scaler.fit(X_tr[:-5])  # except onehot encodings
            joblib.dump(
                self.scaler, 
                f'{self.root_path}/{self.NIA_work}_NIA_train_' \
                f'{self.port}_{year}_scaler_YearSplit.pkl'
            )

            # now save it.
            joblib.dump(self.scaler.transform(X_tr), 
                f'{self.root_path}/{self.NIA_work}_processed_X_train_{year}_YearSplit.pkl')
            joblib.dump(y_tr, 
                f'{self.root_path}/{self.NIA_work}_processed_y_train_{year}_YearSplit.pkl')
            joblib.dump(self.scaler.transform(X_val), 
                f'{self.root_path}/{self.NIA_work}_processed_X_val_{year}_YearSplit.pkl')
            joblib.dump(y_val, 
                f'{self.root_path}/{self.NIA_work}_processed_y_val_{year}_YearSplit.pkl')
            joblib.dump(self.scaler.transform(X_te), 
                f'{self.root_path}/{self.NIA_work}_processed_X_test_{year}_YearSplit.pkl')
            joblib.dump(y_te, 
                f'{self.root_path}/{self.NIA_work}_processed_y_test_{year}_YearSplit.pkl')

        # when saved pre-processed file exist
        self.X = joblib.load(
                f'{self.root_path}/{self.NIA_work}_processed_X_{self.flag}_{year}_YearSplit.pkl')
        self.y = joblib.load(
                f'{self.root_path}/{self.NIA_work}_processed_y_{self.flag}_{year}_YearSplit.pkl')
        self.scaler = joblib.load(  # train 기간에 대해 맞춰진 것
                f'{self.root_path}/{self.NIA_work}_NIA_train_{self.port}_{year}_scaler_YearSplit.pkl')
        

    def __getitem__(self, index):
        return self.X[index, ...], self.y[index, ...]
    
    
    def __len__(self):
        return len(self.X)
        

    def inverse_transform(self, data, is_dnn=False):
        return self.scaler.inverse_transform(data, is_dnn)
