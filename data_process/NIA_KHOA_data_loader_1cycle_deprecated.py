import os
from tkinter.messagebox import NO
import pandas as pd
import joblib
from sklearn.semi_supervised import SelfTrainingClassifier

from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

class Dataset_NIA_KHOA(Dataset):
    def __init__(self, root_path, data, port=None, flag='train', size=None, 
                 features='M', data_path='', scale=True):
    
        if size == None:
            self.seq_len = 2*8*1  # lagging len !!!! 모델에 2^n 제곱 길이만 들어갈 수 있으므로 10분간격 시간단위(6의배수)는 불가능
            self.pred_len = 8*1  # !!!! 모델에 2^n 제곱 길이만 들어갈 수 있으므로 10분간격 시간단위(6의배수)는 불가능
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        self.features = features
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.data = data
        self.port = port
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler() 

        if self.data == 'KHOA':
            if self.set_type != 2:  # when train, validation set
                df_raw = joblib.load(os.path.join(self.root_path, self.data_path))
            else:
                df_raw = pd.read_csv(os.path.join(self.root_path, 'foggen_PTDJ_old_test.csv')) # , encoding='cp949'
            # convolution 연산때문에 input feature size 제한이 있을 수 있음
            df_raw = df_raw['x'].dropna()
            # df_y = df_raw['y']
            # 10분 간격임. 1분간격보다 성능 좋다고 들은 듯?

            # features 만큼 column 가져옴
            if self.features=='S':
                df_data = df_raw[['vis']]
            else:
                df_data = df_raw
            # tr, val, test 각각 범위 지정해서 추출하는 방식
            itv = len(df_data)//10
            border1s = [0, itv*8, 0]  # 시작~처음 80%: train , 나중 20%: val
            border2s = [itv*8, len(df_data), len(df_data)]  # 시작~처음 80%: train , 나중 20%: val
            
            #TODO 기간을 parameter로 받아서 나누는 방식 추가
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            # input feature scaling
            if self.scale:
                print(f'process feature scaling : {self.data}_{self.port}')
                if self.set_type != 2:
                    bordered_data = df_data[border1s[0]:border2s[0]]
                    self.scaler.fit(bordered_data.values)
                    data = self.scaler.transform(df_data.values)
                    joblib.dump(self.scaler, f'{self.root_path}/{self.data}_{self.port}_train_scaler.pkl')
                else:
                    self.scaler = joblib.load(f'{self.root_path}/{self.data}_{self.port}_train_scaler.pkl')
                    data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values
            
            self.data = data[border1:border2, :]  
            # TODO:# input seq 길이 4 : lagging 3까지 써야함. data 행렬을 traspose하기 ... ?? 아냐. feature 수 자체를 바꿔야하나
            # FIXME: 미루고, lagging code from KHOA 받고 나서 lagging 8로 만든 파일을 내가 불러와서 쓰자. 

        elif self.data == 'NIA':
            # FIXME: json file read and extract information from obs-qc file
            # 위치 별 입사각도 빼기
            # 위치 별 one-hot encoding 넣기

            # df_raw = pd.read_csv('./datasets/ETT-data/ETTh1.csv')
            if self.set_type != 2:  # not test
                df_raw_nia = pd.read_csv(os.path.join(self.root_path, self.data_path))
                df_raw_nia = df_raw_nia[df_raw_nia['이안류라벨'].notna()]
            else:
                df_raw_nia = pd.read_csv(os.path.join(self.root_path, 'foggen_PTDJ_old_test.csv')) # , encoding='cp949'
            df_raw_nia = df_raw_nia.drop(columns=['Unnamed: 0', '지수'])

            # features 만큼 column 가져옴
            if self.features=='M':
                df_data = df_raw_nia[df_raw_nia.columns[1:]]  # FIXME !! timestramp 점프하는 구간을 처리 안함 : 동헌씨의 new_prep.py보고 lagging style로 고치기
                df_data = df_data.fillna(method='ffill')  # 13 features + 1 ylabel
            elif self.features=='S':
                df_data = df_raw_nia[['vis', 'y']]
            # tr, val, test 각각 범위 지정해서 추출하는 방식
            # itv = len(df_data)//10  # 10분 간격
            
            # FIXME 개월 단위 split
            itv = len(df_data)//10
            border1s = [0, itv*8, 0]  # 시작~처음 80%: train , 나중 20%: val
            border2s = [itv*8, len(df_data), len(df_data)]  # 시작~처음 80%: train , 나중 20%: val
            #TODO 기간을 parameter로 받아서 나누는 방식 추가
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            
            # input feature scaling
            if self.scale:
                print(f'process feature scaling : {self.data}_{self.port}')
                if self.set_type != 2:
                    bordered_data = df_data[border1s[0]:border2s[0]]  # train sample로만 scaling
                    self.scaler.fit(bordered_data.values)  # standardscaler
                    data = self.scaler.transform(df_data.values)  # scaling을 모든 tr, val set에 적용
                    joblib.dump(self.scaler, f'{self.root_path}/NIA_train_{self.port}_scaler.pkl')
                else:
                    self.scaler = joblib.load(f'{self.root_path}/NIA_train_{self.port}_scaler.pkl')
                    data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values
            
            self.data = data[border1:border2, :]  # 13 features + 1 label


    def __getitem__(self, index):
        if self.data == 'KHOA':
            s_begin = index
            s_end = s_begin + self.seq_len
            r_end = s_end + self.pred_len

            seq_x = self.data[s_begin:s_end]  # input : 13 features + 1 label of input time
            seq_y = self.data[s_end:r_end]  # output : 13 features + 1 label of output time

            return seq_x, seq_y
            
        else:
            # input sequence length 와 ylabel sequence length 는 다른 상황.
            # input sequence : lagging 적용, ylabel seq length : 관심있는 기간만큼의 길이
            s_begin = index
            s_end = s_begin + self.seq_len
            r_end = s_end + self.pred_len

            seq_x = self.data[s_begin:s_end]  # input : 13 features + 1 label of input time
            seq_y = self.data[s_end:r_end]  # output : 13 features + 1 label of output time

            return seq_x, seq_y


    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len+ 1
        # - self.pred_len 안하면 마지막 batch의 특정 index부터 seq_y의 길이가 점점 짧아지게 추출되는 것을 볼 수 있음

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)