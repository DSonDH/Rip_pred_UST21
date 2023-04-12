import pandas as pd
from prophet import Prophet

df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
df.head()

# We fit the model by instantiating a new Prophet object. 
# Any settings to the forecasting procedure are passed into the constructor. 
# Then you call its fit method and pass in the historical dataframe. 
# Fitting should take 1-5 seconds.
m = Prophet()
m.fit(df)

# Predictions are then made on a dataframe with a column ds containing 
# the dates for which a prediction is to be made. 
# You can get a suitable dataframe that extends into the future 
# a specified number of days using the helper method 
# Prophet.make_future_dataframe. 
# By default it will also include the dates from the history, 
# so we will see the model fit as well.
future = m.make_future_dataframe(periods=365)
future.tail()

# The predict method will assign each row in future a predicted value 
# which it names yhat. If you pass in historical dates, 
# it will provide an in-sample fit. The forecast object here is a new dataframe 
# that includes a column yhat with the forecast, as well as columns for 
# components and uncertainty intervals.
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# You can plot the forecast by calling the 
# Prophet.plot method and passing in your forecast dataframe.
fig1 = m.plot(forecast)

# If you want to see the forecast components, you can use the 
# Prophet.plot_components method. By default you’ll see the 
# trend, yearly seasonality, and weekly seasonality of the time series. 
# If you include holidays, you’ll see those here, too.
fig2 = m.plot_components(forecast)
