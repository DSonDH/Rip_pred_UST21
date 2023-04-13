#%%
# %matplotlib inline

#%%
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from io import BytesIO
import itertools


# Register converters to avoid warnings
pd.plotting.register_matplotlib_converters()
plt.rc("figure", figsize=(16,8))
plt.rc("font", size=14)

#%% Dataset
friedman2 = requests.get('https://www.stata-press.com/data/r12/friedman2.dta').content
data = pd.read_stata(BytesIO(friedman2))
data.index = data.time
data.index.freq = "QS-OCT"

# Variables
endog = data.loc['1959':'1981', 'consump']
exog = sm.add_constant(data.loc['1959':'1981', 'm2'])
nobs = endog.shape[0]

# Fit the model
#TODO: order 3개 성부 grid search
p = range(0, 3)
d = range(0, 3)
q = range(0, 3)

pdqs = list(itertools.product(p, d, q))
best_aic = np.inf

for pdq in pdqs:
    model = sm.tsa.statespace.SARIMAX(endog.loc[:'1978-01-01'], 
                                    exog=exog.loc[:'1978-01-01'], 
                                    order=pdq
                                )
    fit_res = model.fit(disp=False, maxiter=250)
    if fit_res.aic < best_aic:
        best_aic = fit_res.aic
        best_pdq = pdq
        best_model = model
        best_fit_res = fit_res

'''
order : iterable or iterable of iterables, optional
    The (p,d,q) order of the model for the number of AR parameters,
    differences, and MA parameters. `d` must be an integer
    indicating the integration order of the process, while
    `p` and `q` may either be an integers indicating the AR and MA
    orders (so that all lags up to those orders are included) or else
    iterables giving specific AR and / or MA lags to include. Default is
    an AR(1) model: (1,0,0).
seasonal_order : iterable, optional
    The (P,D,Q,s) order of the seasonal component of the model for the
    AR parameters, differences, MA parameters, and periodicity.
    `D` must be an integer indicating the integration order of the process,
    while `P` and `Q` may either be an integers indicating the AR and MA
    orders (so that all lags up to those orders are included) or else
    iterables giving speci1fic AR and / or MA lags to include. `s` is an
    integer giving the periodicity (number of periods in season), often it
    is 4 for quarterly data or 12 for monthly data. Default is no seasonal
    effect.
trend : str{'n','c','t','ct'} or iterable, optional
    Parameter controlling the deterministic trend polynomial :math:`A(t)`.
    Can be specified as a string where 'c' indicates a constant (i.e. a
    degree zero component of the trend polynomial), 't' indicates a
    linear trend with time, and 'ct' is both. Can also be specified as an
    iterable defining the non-zero polynomial exponents to include, in
    increasing order. For example, `[1,1,0,1]` denotes
    :math:`a + bt + ct^3`. Default is to not include a trend component.
seasonal_order: Default is no seasonal effect. 이안류 개별 instance들은 주기성
    없으니까 이 옵션은 꺼야함 !!!
'''

print(best_fit_res.summary())

#%% In-sample one-step-ahead predictions
res = best_model.filter(best_fit_res.params)

predict = res.get_prediction()
predict_ci = predict.conf_int()

# Dynamic predictions
predict_dy = res.get_prediction(dynamic='1978-01-01')
predict_dy_ci = predict_dy.conf_int()


#%% Graph
fig, ax = plt.subplots(figsize=(9,4))
npre = 4
ax.set(title='Personal consumption', xlabel='Date', ylabel='Billions of dollars')

# Plot data points
data.loc['1977-07-01':, 'consump'].plot(ax=ax, style='o', label='Observed')

# Plot predictions
predict.predicted_mean.loc['1977-07-01':].plot(ax=ax, style='r--', label='One-step-ahead forecast')
ci = predict_ci.loc['1977-07-01':]
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='r', alpha=0.1)
predict_dy.predicted_mean.loc['1977-07-01':].plot(ax=ax, style='g', label='Dynamic forecast (1978)')
ci = predict_dy_ci.loc['1977-07-01':]
ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='g', alpha=0.1)

legend = ax.legend(loc='lower right')
