import json
import os
# # 같은 feature에 다른 timepoint를 따로 계산함... reverse도 제대로 안됨
import warnings

import joblib
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from utils.tools import StandardScaler

warnings.filterwarnings('ignore')

def add_one_hot(df, site, site_names = ['DC', 'HD', 'JM', 'NS', 'SJ']):
    one_hot = np.zeros(len(site_names))
    one_hot[site_names.index(site)] = 1
    df[site_names] = one_hot
    return df


class Dataset_NIA(Dataset):
    def __init__(self, root_path, NIA_work, data, port=None, flag='train', 
                 size=None, data_path='', args=None):
    
        if size == None:
            self.seq_len = 2*8*2  # lagging len !!!! 모델에 2^n 제곱 길이만 들어갈 수 있으므로 10분간격 시간단위(6의배수)는 불가능
            self.pred_len = 8*2  # !!!! 모델에 2^n 제곱 길이만 들어갈 수 있으므로 10분간격 시간단위(6의배수)는 불가능
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.NIA_work = NIA_work
        self.mode = flag
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.data = data
        self.port = port
        self.args = args
        self.save_meta_csv = False  # FIXME:

        self.__read_data__()

    def load_df(self, csv_pth, site, year, angle_inci_beach):
        '''nan filling 작업, wave direction 보정 작업'''

        df = pd.read_csv(f'{csv_pth}/{site}-total.csv', encoding='euc-kr')
        df = df.iloc[:,1:]
        df['ob time'] = pd.to_datetime(df['ob time'])
        df = df.set_index('ob time')
        df = df.fillna(method='ffill', limit=2).fillna(method='bfill', limit=2)
        df['wave direction'] -= angle_inci_beach
        return df
                    

    def __read_data__(self):
        """
        train, val, test 각각 dataloader 실행되는 구조.
        각각 실행 될때마다 train 구하고, val구하고 test구하는 방식
        
        all_site 모아서 scaler 부터 구함
        all_site 분리해서 샘플 추출함

        In case of training dataset extacting phase,

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
        
        Finalize final train dataset
        get index for training loop using final train dataset
        """

        #TODO : 다른 시계열 연구에서는 normalization 어떻게 하는지, cross validaton split어찌 하는 지 조사

        self.scaler = StandardScaler() #MinMaxScaler()
        site_names = ['DC', 'HD', 'JM', 'NS', 'SJ']
        angle_inci_beach = [245, 178, 175, 47, 142]
        year = self.args.year

        if self.data == 'NIA':
            if not os.path.exists(
                f'{self.root_path}/{self.NIA_work}_processed_X_{self.mode}_{year}_csvBase.pkl'):
                print(f'no processed {self.mode} file. start data preprocessing!! \n')
                
                csv_pth = f'{self.root_path}/obs_qc_100p'

                # Step1: merge train set each site and get scaler first
                # TODO: 엄밀하게는 tr/val/te기간 나눠서 tr기간에 대해서만 scaler fit해야함.
                if self.mode == 'train':
                    for i, site in enumerate(site_names):
                        df = self.load_df(csv_pth, site, year, angle_inci_beach[i])
                        df_all = df if i==0 else pd.concat([df_all, df])
                    self.scaler.fit(df_all)
                    joblib.dump(
                        self.scaler, 
                        f'{self.root_path}/{self.NIA_work}_NIA_train_{self.port}_{year}_scaler_csvBase.pkl'
                    )

                # Step2: loop each site to get instance and split tr/val/te = 8:1:1
                X = []
                y = []
                for i, site in enumerate(site_names):
                    print(f' ## processing {site} for {self.mode}')

                    df = self.load_df(csv_pth, site, year, angle_inci_beach[i])

                    ''' csv : 매년 6월01일 00시 ~ 8월 31일 23시 55분까지 5분간격.
                        Way1: tr, val, test 기간 따로 정하여 나누는 방법.
                        Way2: 8:1:1 timestamp 갯수로 나누는 방법.  <--- Current Way
                    '''
                    if self.save_meta_csv:
                        # 이안류 라벨, 관측자료 nan 제거함 : 관측자료 결측은 일단 무시
                        # noNanMask = ~df['RipLabel'].isna()
                        # df2 = df[noNanMask]
                        df2 = df.dropna(axis=0)
                        df2.to_csv(f'datasets/NIA/meta_csv/{site}_dropNaN.csv',
                                   encoding='euc-kr')
                    
                    # print label ratio
                    n1 = sum(df['RipLabel'] == 1)
                    n0 = sum(df['RipLabel'] == 0)
                    print(f'N_raw instance = {n1 + n0}')
                    print(f'N_0 instance = {n0}, N_1 instance = {n1}')
                    print(f'N1 / N_0 ratio = {n1 / (n0 + n1) * 100:.2f} \n')

                    # scaler
                    if self.mode != 'train':
                        self.scaler = joblib.load(
                            f'{self.root_path}/{self.NIA_work}_NIA_train_{self.port}_{year}_scaler_csvBase.pkl')
                    df = self.scaler.transform(df)

                    instance_list = []
                    origin_idx_list = []
                    N_nan = 0
                    for x_start in range(self.seq_len, len(df)):
                        y_end = x_start + self.seq_len + self.pred_len
                        # 일단 X, y합쳐서 뽑고, 나중에 분리
                        Xy_instance = df.iloc[x_start:y_end, :]
                        if not Xy_instance.isnull().values.any():
                            instance_list.append(Xy_instance.values)
                            origin_idx_list.append(x_start)
                        else:
                            N_nan += 1

                    num_instance = len(instance_list)
                    itv = num_instance//10
                    Xy_full = np.array(instance_list)

                    # val, test set 갯수 정확히 같게 setting
                    if (len(Xy_full[itv * 8:]) % 2) == 1:
                        idx_tr = itv * 8 + 1
                        N_valtr = len(Xy_full[idx_tr :])
                        idx_val = idx_tr + N_valtr // 2
                    elif len(Xy_full[itv * 8 : itv * 9]) != len(Xy_full[itv * 9:]):
                        idx_tr = itv * 8
                        N_valtr = len(Xy_full[idx_tr :])
                        idx_val = idx_tr + N_valtr // 2
                    else:
                        idx_tr = itv * 8
                        idx_val = itv * 9

                    if self.mode == 'train':
                        Xy = Xy_full[: idx_tr]
                    elif self.mode  == 'val':
                        Xy = Xy_full[idx_tr : idx_val]
                    else:
                        Xy = Xy_full[idx_val :]

                    if self.save_meta_csv:
                        idx_org_tr = origin_idx_list[: idx_tr]
                        idx_org_val = origin_idx_list[idx_tr : idx_val]
                        idx_org_te = origin_idx_list[idx_val:]

                        df_rvrs = self.scaler.inverse_transform(df)
                        df_tr = df_rvrs.iloc[idx_org_tr, :]
                        df_val = df_rvrs.iloc[idx_org_val, :]
                        df_te = df_rvrs.iloc[idx_org_te, :]
                        
                        df_tr.to_csv(f'datasets/NIA/meta_csv/{site}_train.csv',
                                     encoding='euc-kr')
                        df_val.to_csv(f'datasets/NIA/meta_csv/{site}_val.csv',
                                     encoding='euc-kr')
                        df_te.to_csv(f'datasets/NIA/meta_csv/{site}_test.csv',
                                     encoding='euc-kr')


                    len_tr = len_val = len_te = 0
                    len_tr += len(Xy_full[: idx_tr])
                    len_val += len(Xy_full[idx_tr : idx_val])
                    len_te += len(Xy_full[idx_val :])

                    # add onehot
                    n_sites = len(site_names)
                    onehot = np.zeros((Xy.shape[0], Xy.shape[1], n_sites))
                    onehot[:,:,i] = 1
                    Xy = np.concatenate((Xy, onehot), axis=2)

                    # merge X, y seperately
                    X.append(Xy[:, :self.seq_len, :])
                    y.append(Xy[:, self.seq_len:, :])
                    print(f'NoNan : Nan of {site} = {num_instance} : {N_nan}')
                    print(f'[{site}] tr:val:te = {len_tr} : {len_val} : {len_te}')


                # Step3: merge array finally
                X = np.concatenate(X, axis=0)
                y = np.concatenate(y, axis=0)

                # now save it.
                joblib.dump(X, 
                    f'{self.root_path}/{self.NIA_work}_processed_X_{self.mode}_{year}_csvBase.pkl')
                joblib.dump(y, 
                    f'{self.root_path}/{self.NIA_work}_processed_y_{self.mode}_{year}_csvBase.pkl')

            # when saved pre-processed file exist
            else:
                print(f'I found processed {self.mode} file. skip data preprocessing!!\n')
                X = joblib.load(
                        f'{self.root_path}/{self.NIA_work}_processed_X_{self.mode}_{year}_csvBase.pkl')
                y = joblib.load(
                        f'{self.root_path}/{self.NIA_work}_processed_y_{self.mode}_{year}_csvBase.pkl')
                self.scaler = joblib.load(  # train mode만 불러야 함
                        f'{self.root_path}/{self.NIA_work}_NIA_train_{self.port}_{year}_scaler_csvBase.pkl')
            
            self.X = X
            self.y = y


    def __getitem__(self, index):
        return self.X[index, ...], self.y[index, ...]
    
    
    def __len__(self):
        return len(self.X)
        
    def inverse_transform(self, data, is_dnn=False):
        return self.scaler.inverse_transform(data, is_dnn)