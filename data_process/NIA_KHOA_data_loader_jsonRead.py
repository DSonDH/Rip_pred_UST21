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


class Dataset_NIA_KHOA(Dataset):
    def __init__(self, root_path, data, port=None, flag='train', size=None, 
                 features='M', data_path='', scale=True):
    
        if size == None:
            self.seq_len = 2*8*2  # lagging len !!!! 모델에 2^n 제곱 길이만 들어갈 수 있으므로 10분간격 시간단위(6의배수)는 불가능
            self.pred_len = 8*2  # !!!! 모델에 2^n 제곱 길이만 들어갈 수 있으므로 10분간격 시간단위(6의배수)는 불가능
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.mode = flag
        self.set_type = type_map[flag]
        self.features = features
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.data = data
        self.port = port
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler() #MinMaxScaler()
        n_features = 11 if self.features=='M'else 1
        site_names = ['DC', 'HD', 'JM', 'NS', 'SJ']

        if self.data == 'NIA':
            if not os.path.exists(f'{self.root_path}/ripcurrent_10p/processed_{self.mode}_jsonBase.pkl'):
                print(f'no processed {self.mode} file. start data preprocessing!!')
                ann_pth = f'{self.root_path}/ripcurrent_10p/{self.mode}'
                csv_pth = f'{self.root_path}/obs_qc'
                ann_list = os.listdir(ann_pth)
                
                tmp_site_year = f'??_????'

                num_instance = 0
                for ann in ann_list:
                    with open(f'{ann_pth}/{ann}') as json_file:
                        annot = json.load(json_file)
                    fname = annot['image_info']['file_name']
                    site = fname.split('_')[0]
                    obs_time = '_'.join(fname[:-4].split('_')[2:4])
                    year = obs_time[:4]
                    angle_inci_beach = annot['annotations']['angle_of_incidence_of_the_beach']
                    
                    if tmp_site_year != f'{site}_{year}':  # 매번 csv읽으면 병목 오짐
                        tmp_site_year = f'{site}_{year}'
                        df = pd.read_csv(f'{csv_pth}/{site}_{year}_qc.csv', encoding='euc-kr')
                        df = df.iloc[:,1:]
                        df['ob time'] = pd.to_datetime(df['ob time'])
                        df = df.set_index('ob time')
                        df = df.fillna(method='ffill', limit=2).fillna(method='bfill', limit=2)
                        df['wave direction'] -= angle_inci_beach
                        df = add_one_hot(df, site, site_names)

                        if self.features=='S':
                            df = df['이안류라벨']
                            df = add_one_hot(df, site, site_names)
                
                    # csv : 매년 6월01일 00시 ~ 8월 31일 23시 55분까지 5분간격
                    # 기준 index 찾고, sequence length 만큼 추출
                    obs_time_cnvrt = pd.to_datetime(obs_time, format='%Y%m%d_%H%M%S')
                    idx = df.index.get_loc(obs_time_cnvrt, method='nearest')
                    
                    x_start = idx - self.seq_len + 1
                    y_end = idx + self.pred_len + 1

                    if x_start < 0 or y_end > len(df):
                        # index out of range'
                        continue

                    Xy_instance = df.iloc[x_start:y_end, :]  # 일단 합쳐서 뽑고, 나중에 normalize후에 분리
                    if Xy_instance.isnull().values.any():
                        continue # skip nan 

                    # collect all instances
                    if num_instance == 0:
                        Xy_full = Xy_instance.values
                    else:
                        Xy_full = np.concatenate((Xy_full, Xy_instance.values), axis=0)
                        # 모든 배치를 feature에 맞춰서 세로로 이어붙임 -> normalize 용이함
                    num_instance += 1

                # input feature scaling
                if self.features=='M' and self.scale:
                    if self.mode == 'train':
                        self.scaler.fit(Xy_full)  # MinMaxScaler
                        data = self.scaler.transform(Xy_full)  # scaling을 모든 tr, val set에 적용
                        joblib.dump(self.scaler, f'{self.root_path}/NIA_train_{self.port}_scaler_jsonBase.pkl')
                        data2 = self.scaler.inverse_transform(data)
                    else:
                        self.scaler = joblib.load(f'{self.root_path}/NIA_train_{self.port}_scaler_jsonBase.pkl')
                        data = self.scaler.transform(Xy_full)
                else:
                    data = Xy_full
                
                # save final processed data
                joblib.dump(data, f'{self.root_path}/ripcurrent_10p/processed_{self.mode}_jsonBase.pkl')

            # when saved pre-processed file exist
            else:
                print(f'I found processed {self.mode} file. skip data preprocessing!!')
                data = joblib.load(f'{self.root_path}/ripcurrent_10p/processed_{self.mode}_jsonBase.pkl')
                num_instance = data.shape[0]//(self.seq_len + self.pred_len)
                self.scaler = joblib.load(f'{self.root_path}/NIA_train_{self.port}_scaler_jsonBase.pkl')
            
            # reshape to bachsize x timepoint x numfeatures            
            data2 = data.reshape((num_instance, -1, data.shape[-1]))

            self.X = data2[:,:self.seq_len,:]
            self.y = data2[:,self.seq_len:,:]

    def __getitem__(self, index):
        return self.X[index, ...], self.y[index, ...]
    
    def __len__(self):
        return len(self.X)
        
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)