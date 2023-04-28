import sys
from tqdm.contrib.concurrent import process_map
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

sys.path.append('/home/sdh/rippred/')
from data_process.NIA_data_loader_csvOnly_YearSplit import Dataset_NIA_class


def mmx(feature: pd.Series) -> pd.Series:
    '''minmax scaler'''
    return (feature - np.nanmin(feature)) / (np.nanmax(feature) - np.nanmin(feature))


def draw_and_save(df: pd.DataFrame, save_path: str) -> None:
    """Draw NxN scatter plot using input dataframe and save to the save_path"""

    n = len(df.columns)
    fig, axes = plt.subplots(n, n, sharex=True, sharey=True, figsize=(21, 21))
    for i in range(n):
        for j in range(n):
            if i < j:
                continue
            labels = df.iloc[:, 10]
            rip_mask = (labels == 1)
            feature1 = df.iloc[:, i]
            feature2 = df.iloc[:, j]

            feature1 = mmx(feature1)
            feature2 = mmx(feature2)
            f2_rip, f2_norip = feature2[rip_mask], feature2[~rip_mask]

            # scatter plotting
            axes[i, j].scatter(f2_rip, feature1[rip_mask], color='lightcoral', 
                            s=5, alpha=0.5, zorder=10)
            axes[i, j].scatter(f2_norip, feature1[~rip_mask], color='slategray', 
                            s=5, alpha=0.5, zorder=1)
            
            # x, y label setting
            if j == 0:
                axes[i, j].set_ylabel(df.columns[i], fontsize=13)
            if i == n - 1:
                axes[i, j].set_xlabel(df.columns[j], fontsize=13)

    fig.tight_layout()
    plt.savefig(f'{save_path}', dpi=150)
    plt.close()

# main
draw_type1 = True
csv_path = '/home/sdh/rippred/datasets/NIA/obs_qc_100p'
save_path = '/home/sdh/rippred/result/feature_scatter_plot'
site_names = ['DC', 'HD', 'JM', 'NS', 'SJ']
angle_inci_beach = [245, 178, 175, 47, 142]
featueres = ['rip index', 'wave height', 'wave period', 'tide height', 
             'peak period', 'spectrum factor', 'wave direction', 
             'wave direction factor', 'wind direction', 'wind speed', 
             'RipLabel']

if draw_type1:
    # each tr/val/test split, each site together
    #=> 3 x 5 = 15개 이미지 생성
    # whole year, whole site together

    start_year = '2019'
    for mode, next_year in zip(['train', 'val', 'test'], ['2021', '2022', '2023']):
        for ii, site in enumerate(site_names):
            df = pd.read_csv(f'{csv_path}/{site}-total.csv', encoding='euc-kr')
            df = df.iloc[:, 1:]
            df = df.fillna(method='ffill', limit=2).fillna(method='bfill', limit=2)
            df['wave direction'] -= angle_inci_beach[ii]  # wave direction 보정    
            
            time_selection = (df['ob time'] > start_year) * (df['ob time'] < next_year)
            df_tmp = df[time_selection].drop(columns='ob time')
            draw_and_save(df_tmp, f'{save_path}/all_feature_scatter_plot_{mode}_{site}.jpg')

        start_year = next_year

else:
    # whole year, whole site together
    df_full = []

    for i, site in enumerate(site_names):
        df_tmp = pd.read_csv(f'{csv_path}/{site}-total.csv', encoding='euc-kr')
        df_tmp = df_tmp.iloc[:, 1:]
        df_tmp = df_tmp.drop(columns='ob time')
        df_tmp = df_tmp.fillna(method='ffill', limit=2).fillna(method='bfill', limit=2)
        df_tmp['wave direction'] -= angle_inci_beach[i]  # wave direction 보정    
        df_full = df_tmp if len(df_full) == 0 else pd.concat([df_full, df_tmp])

    draw_and_save(df_full, f'{save_path}/all_feature_scatter_plot.jpg')
