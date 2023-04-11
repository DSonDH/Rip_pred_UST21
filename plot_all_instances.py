from tqdm.contrib.concurrent import process_map
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


root_path = './datasets/NIA/'
site_names = ['DC', 'HD', 'JM', 'NS', 'SJ']
angle_inci_beach = [245, 178, 175, 47, 142]
seq_len = 32
pred_len = 16

featueres = ['rip index', 'wave height', 'wave period', 'tide height', 
             'peak period', 'spectrum factor', 'wave direction', 
             'wave direction factor', 'wind direction', 'wind speed', 
             'RipLabel']


def load_df(csv_pth: str, 
            site: str, 
            angle_inci_beach: int
            ) -> pd.DataFrame:

    df = pd.read_csv(f'{csv_pth}/{site}-total.csv', encoding='euc-kr')
    df = df.iloc[:,1:]
    df['ob time'] = pd.to_datetime(df['ob time'])
    df = df.set_index('ob time')
    df = df.fillna(method='ffill', limit=2).fillna(method='bfill', limit=2)
    df['wave direction'] -= angle_inci_beach  # wave direction 보정
    return df


for year in [2019, 2020, 2021, 2022]:
    csv_pth = f'{root_path}/obs_qc_100p'


    for i, site in enumerate(site_names):

        df = load_df(csv_pth, site, angle_inci_beach[i])
        
        # print label ratio
        n1 = sum(df['RipLabel'] == 1)
        n0 = sum(df['RipLabel'] == 0)
        print(f'N_raw instance = {n1 + n0}')
        print(f'N_0 instance = {n0}, N_1 instance = {n1}')
        print(f'N1 / N_0 ratio = {n1 / (n0 + n1) * 100:.2f} \n')

        instance_list = []
        origin_idx_list = []
        N_nan = 0

        for x_start in range(seq_len, len(df)):
            y_end = x_start + seq_len + pred_len
            # 일단 X, y합쳐서 뽑고, 나중에 분리
            Xy_instance = df.iloc[x_start:y_end, :]
            if not Xy_instance.isnull().values.any():
                instance_list.append(Xy_instance.values)
                origin_idx_list.append(x_start)
            else:
                N_nan += 1

        num_instance = len(instance_list)
        Xy_full = np.array(instance_list)

        X = Xy_full[:, :seq_len, :]
        y = Xy_full[:, seq_len:, :]
        
        # drawing start !!
        for idx in range(0, len(X), 100):
            fig, axs = plt.subplots(nrows=X.shape[-1], figsize=(10, 9))

            for ii, feature in enumerate(featueres):
                x = np.arange(seq_len)
                y = X[idx, ..., ii]

                axs[ii].plot(x, y, color='b', label = f'{feature}')
                # axs[i].set_xlim([pd.to_datetime(f'{year}-05-31'), pd.to_datetime(f'{year}-09-01')])
                axs[ii].tick_params(axis='both', labelsize=10)
                if ii == 10:  # final row
                    axs[ii].set_xticklabels(axs[i].get_xticklabels(), rotation=10)
                    axs[ii].set_ylim([-0.1, 1.1])
                else:
                    axs[ii].set_xticks([])
                axs[ii].legend(loc='upper left', fontsize=10)
                axs[ii].grid(True)

            plt.suptitle(f'{site}_{year}_instance{idx} all features\n', fontsize=15)
            plt.tight_layout()
            plt.savefig(f'./rip_obs_instance_plot_version1/instance{idx}_{site}_{year}.png', dpi=100)
            plt.close()
            
