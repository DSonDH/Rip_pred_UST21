import pandas as pd
import matplotlib.pyplot as plt

site_names = {'Daecheon':'DC', 'Haeundae':'HD', 'Jungmun':'JM', 'Naksan':'NS', 'Songjeong':'SJ'}

angle_inci_beach = {'Daecheon':245, 'Haeundae':178, 'Jungmun':175, 'Naksan':47, 'Songjeong':142}

root_path = '/home/sdh/rippred/datasets/NIA'
csv_pth = f'{root_path}/obs_qc_100p'

featueres = ['rip index', 'wave height', 'wave period', 'tide height', 
             'peak period', 'spectrum factor', 'wave direction', 
             'wave direction factor', 'wind direction', 'wind speed', 
             '이안류라벨']
years = [2019, 2020, 2021, 2022]

plot_version = 1
# version 1 : all feature in each year
# versioni 2 : all year in each feature

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
            
            # drawing start !!
            fig, axs = plt.subplots(nrows=len(featueres), figsize=(25, 18))

            for i, feature in enumerate(featueres):
                roi = df[time_mask][feature]
                x = roi.index
                y = roi.values

                axs[i].plot(x, y,
                        color='b', label = f'{site}_{year}_{feature}')
                
                axs[i].tick_params(axis='both', labelsize=10)
                if i == 10:
                    axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=10)
                else:
                    axs[i].set_xticks([])
                axs[i].legend(loc='right', fontsize=12)
                axs[i].grid(True)
            
            plt.suptitle(f'{site}_{year}_all features', fontsize=24)
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
                y = roi.values

                axs[i].plot(x, y,
                        color='b', label = f'{site}_{year}_{feature}')
                
                axs[i].tick_params(axis='both', labelsize=10)
                axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=10)
                axs[i].legend(loc=9, fontsize=7, ncol=4)
                axs[i].grid(True)
            
            plt.tight_layout()
            plt.savefig(f'./rip_obs_timeseries_plot_version2/NIA_plot_{site}_{feature}.png', dpi=300)
            plt.close()
