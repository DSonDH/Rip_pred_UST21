import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

site_names = {'Daecheon':'DC', 'Haeundae':'HD', 'Jungmun':'JM', 'Naksan':'NS', 'Songjeong':'SJ'}

angle_inci_beach = {'Daecheon':245, 'Haeundae':178, 'Jungmun':175, 'Naksan':47, 'Songjeong':142}

root_path = '/home/sdh/rippred/datasets/NIA'
csv_pth = f'{root_path}/obs_qc_100p'

featueres = ['rip index', 'wave height', 'wave period', 'tide height', 
             'peak period', 'spectrum factor', 'wave direction', 
             'wave direction factor', 'wind direction', 'wind speed', 
             'RipLabel']
years = [2019, 2020, 2021, 2022]

plot_version = 1
# version 1 : all feature in each year
# versioni 2 : all year in each feature

# plt.rcParams['font.family'] = 'NanumGothic'
# mpl.rcParams['axes.unicode_minus'] = False

"""
print(mpl.matplotlib_fname())
print(mpl.__file__)

import matplotlib.font_manager as fm
f = [f.name for f in fm.fontManager.ttflist]
print(f)

# plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams['font.family'] = 'NanumGothic'
print(plt.rcParams['font.family'])
mpl.rcParams['axes.unicode_minus'] = False

# plt.text(0.3, 0.3, '한글', size = 100)
"""

if plot_version == 1:
    
    # version1 : plot for each site, each years, all features
    for site in site_names.keys():
        df = pd.read_csv(f'{csv_pth}/{site_names[site]}-total.csv', encoding='euc-kr')
        df = df.iloc[:,1:]
        df['ob time'] = pd.to_datetime(df['ob time'])
        df = df.set_index('ob time')
        df['wave direction'] -= angle_inci_beach[site]
        
        for year in years:
            time_mask = (df.index < pd.to_datetime(f'{year + 1}-01-01')) * \
                        (df.index >= pd.to_datetime(f'{year}-01-01'))
            rip_mask = df[time_mask]['RipLabel'] == 1

            # serach all rip current period of a year
            x = df[time_mask].index
            rip_period = []
            start = None
            end = None
            for time in x:
                if df.loc[time]['RipLabel'] == 1 and start == None:
                    start = str(time)
                elif start != None and df.loc[time]['RipLabel'] != 1:
                    end = str(time)
                    rip_period.append((start, end))    
                    start = None
                    end = None

            # drawing start !!
            fig, axs = plt.subplots(nrows=len(featueres), figsize=(30, 18))

            for i, feature in enumerate(featueres):
                roi = df[time_mask][feature]
                x = roi.index
                y = roi.values

                axs[i].plot(x, y, color='b', label = f'{feature}')
                axs[i].set_xlim([pd.to_datetime(f'{year}-05-31'), pd.to_datetime(f'{year}-09-01')])
                axs[i].tick_params(axis='both', labelsize=20)
                if i == 10:  # final row
                    axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=10)
                else:
                    axs[i].set_xticks([])
                axs[i].legend(loc='upper left', fontsize=20)
                axs[i].grid(True)

                # 모든 시기 이안류색깔 표시
                for start, end in rip_period:
                    axs[i].axvspan(start, end, alpha=0.2, color='red')

            plt.suptitle(f'{site}_{year}_all features\n', fontsize=30)
            plt.tight_layout()
            plt.savefig(f'./rip_obs_timeseries_plot_version1/NIA_plot_{site}_{year}.png', dpi=300)
            plt.close()


elif plot_version == 2:

    # version2 : plot for each site, each features, all years
    for site in site_names.keys():
        df = pd.read_csv(f'{csv_pth}/{site_names[site]}-total.csv', encoding='euc-kr')
        df = df.iloc[:,1:]
        df['ob time'] = pd.to_datetime(df['ob time'])
        df = df.set_index('ob time')
        df['wave direction'] -= angle_inci_beach[site]
        
        for feature in featueres:

            fig, axs = plt.subplots(nrows=len(years))

            for i, year in enumerate(years):
                
                time_mask = (df.index < pd.to_datetime(f'{year + 1}-01-01')) * \
                            (df.index >= pd.to_datetime(f'{year}-01-01'))
                        
                roi = df[time_mask][feature]
                x = roi.index
                y = roi.valuesN

                axs[i].plot(x, y,
                        color='b', label = f'{site}_{year}_{feature}')
                
                axs[i].tick_params(axis='both', labelsize=10)
                axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=10)
                axs[i].legend(loc=9, fontsize=7, ncol=4)
                axs[i].grid(True)
            
            plt.tight_layout()
            plt.savefig(f'./rip_obs_timeseries_plot_version2/NIA_plot_{site}_{feature}.png', dpi=300)
            plt.close()
