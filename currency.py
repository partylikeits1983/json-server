import pandas as pd
from pandas_datareader import data, wb  
import datetime
import numpy as np 
from fbprophet import Prophet


start = datetime.datetime(2016,6,1)
end = datetime.date.today()
df = data.DataReader("RUB=X", "yahoo", start, end)
df['unix'] = df.index
df['unix'] = df.unix.apply(lambda x : (x-datetime.datetime(1970,1,1)).total_seconds())
df.drop(df.columns.difference(['Date','Open', 'High', 'Low', 'Close', 'unix']), 1, inplace=True)
df = df.round(4)
df.to_json(r'/home/ubuntu/Desktop/RUBOHLC.json', orient="values")
df.to_csv('/home/ubuntu/Desktop/RUB.csv')
graph = pd.read_csv('/home/ubuntu/Desktop/RUB.csv')
graph = graph[['Date','Close']]
graph.set_index('Date', inplace=True, drop=True) 
graph.plot(grid = True)
stock_return = graph.apply(lambda x: x / x[0])
stock_return.head()
stock_return.plot(grid = True).axhline(y = 1, color = "black", lw = 2)
stock_change = graph.apply(lambda x: np.log(x) - np.log(x.shift(1))) # shift moves dates back by 1.
stock_change.head()
stock_change.plot(grid = True).axhline(y = 0, color = "black", lw = 2)
graph["20d"] = np.round(graph["Close"].rolling(window = 20, center = False).mean(), 2)
df = pd.DataFrame()
df['ds'] = stock_return.index
df['y']=graph['Close'].apply(lambda x: np.log(x)).values
df.tail()
m0 = Prophet(yearly_seasonality=True)
#m0 = Prophet(daily_seasonality=False)
m0.fit(df)
#n_add = 365 - len()
n_add = 100
future = m0.make_future_dataframe(periods=n_add, freq='D') 
future.tail(10)
forecast = m0.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()
forcast = m0.plot(forecast, ylabel='$\ln($stock_return$)$');
trend = m0.plot_components(forecast);
forecast['exp_yhat'] = np.exp(forecast['yhat'])
forecast['exp_yhat_lower'] = np.exp(forecast['yhat_lower'])
forecast['exp_yhat_upper'] = np.exp(forecast['yhat_upper'])
col_list = ['ds','exp_yhat', 'exp_yhat_lower', 'exp_yhat_upper']
df = forecast[col_list]
unix = df.ds.apply(lambda x : (x-datetime.datetime(1970,1,1)).total_seconds())
df.drop('ds', axis=1, inplace=True)
df = df.join(unix)
df = df.round(2)
df.tail()
df.to_json(r'/home/ubuntu/Desktop/RUB.json', orient="values")



start = datetime.datetime(2016,6,1)
end = datetime.date.today()
df = data.DataReader("DX-Y.NYB", "yahoo", start, end)
df['unix'] = df.index
df['unix'] = df.unix.apply(lambda x : (x-datetime.datetime(1970,1,1)).total_seconds())
df.drop(df.columns.difference(['Date','Open', 'High', 'Low', 'Close', 'unix']), 1, inplace=True)
df = df.round(4)
df.to_json(r'/home/ubuntu/Desktop/DXYOHLC.json', orient="values")
df.to_csv('/home/ubuntu/Desktop/DXY.csv')
graph = pd.read_csv('/home/ubuntu/Desktop/DXY.csv')
graph = graph[['Date','Close']]
graph.set_index('Date', inplace=True, drop=True) 
graph.plot(grid = True)
stock_return = graph.apply(lambda x: x / x[0])
stock_return.head()
stock_return.plot(grid = True).axhline(y = 1, color = "black", lw = 2)
stock_change = graph.apply(lambda x: np.log(x) - np.log(x.shift(1))) # shift moves dates back by 1.
stock_change.head()
stock_change.plot(grid = True).axhline(y = 0, color = "black", lw = 2)
graph["20d"] = np.round(graph["Close"].rolling(window = 20, center = False).mean(), 2)
df = pd.DataFrame()
df['ds'] = stock_return.index
df['y']=graph['Close'].apply(lambda x: np.log(x)).values
df.tail()
m0 = Prophet(yearly_seasonality=True)
#m0 = Prophet(daily_seasonality=False)
m0.fit(df)
#n_add = 365 - len()
n_add = 100
future = m0.make_future_dataframe(periods=n_add, freq='D') 
future.tail(10)
forecast = m0.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()
forcast = m0.plot(forecast, ylabel='$\ln($stock_return$)$');
trend = m0.plot_components(forecast);
forecast['exp_yhat'] = np.exp(forecast['yhat'])
forecast['exp_yhat_lower'] = np.exp(forecast['yhat_lower'])
forecast['exp_yhat_upper'] = np.exp(forecast['yhat_upper'])
col_list = ['ds','exp_yhat', 'exp_yhat_lower', 'exp_yhat_upper']
df = forecast[col_list]
unix = df.ds.apply(lambda x : (x-datetime.datetime(1970,1,1)).total_seconds())
df.drop('ds', axis=1, inplace=True)
df = df.join(unix)
df = df.round(2)
df.tail()
df.to_json(r'/home/ubuntu/Desktop/DXY.json', orient="values")




start = datetime.datetime(2016,6,1)
end = datetime.date.today()
df = data.DataReader("JPY=X", "yahoo", start, end)
df['unix'] = df.index
df['unix'] = df.unix.apply(lambda x : (x-datetime.datetime(1970,1,1)).total_seconds())
df.drop(df.columns.difference(['Date','Open', 'High', 'Low', 'Close', 'unix']), 1, inplace=True)
df = df.round(4)
df.to_json(r'/home/ubuntu/Desktop/JPYOHLC.json', orient="values")
df.to_csv('/home/ubuntu/Desktop/JPY.csv')
graph = pd.read_csv('/home/ubuntu/Desktop/JPY.csv')
graph = graph[['Date','Close']]
graph.set_index('Date', inplace=True, drop=True) 
graph.plot(grid = True)
stock_return = graph.apply(lambda x: x / x[0])
stock_return.head()
stock_return.plot(grid = True).axhline(y = 1, color = "black", lw = 2)
stock_change = graph.apply(lambda x: np.log(x) - np.log(x.shift(1))) # shift moves dates back by 1.
stock_change.head()
stock_change.plot(grid = True).axhline(y = 0, color = "black", lw = 2)
graph["20d"] = np.round(graph["Close"].rolling(window = 20, center = False).mean(), 2)
df = pd.DataFrame()
df['ds'] = stock_return.index
df['y']=graph['Close'].apply(lambda x: np.log(x)).values
df.tail()
m0 = Prophet(yearly_seasonality=True)
#m0 = Prophet(daily_seasonality=False)
m0.fit(df)
#n_add = 365 - len()
n_add = 100
future = m0.make_future_dataframe(periods=n_add, freq='D') 
future.tail(10)
forecast = m0.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()
forcast = m0.plot(forecast, ylabel='$\ln($stock_return$)$');
trend = m0.plot_components(forecast);
forecast['exp_yhat'] = np.exp(forecast['yhat'])
forecast['exp_yhat_lower'] = np.exp(forecast['yhat_lower'])
forecast['exp_yhat_upper'] = np.exp(forecast['yhat_upper'])
col_list = ['ds','exp_yhat', 'exp_yhat_lower', 'exp_yhat_upper']
df = forecast[col_list]
unix = df.ds.apply(lambda x : (x-datetime.datetime(1970,1,1)).total_seconds())
df.drop('ds', axis=1, inplace=True)
df = df.join(unix)
df = df.round(2)
df.tail()
df.to_json(r'/home/ubuntu/Desktop/JPY.json', orient="values")





start = datetime.datetime(2016,6,1)
end = datetime.date.today()
df = data.DataReader("GBPUSD=X", "yahoo", start, end)
df['unix'] = df.index
df['unix'] = df.unix.apply(lambda x : (x-datetime.datetime(1970,1,1)).total_seconds())
df.drop(df.columns.difference(['Date','Open', 'High', 'Low', 'Close', 'unix']), 1, inplace=True)
df = df.round(4)
df.to_json(r'/home/ubuntu/Desktop/GBPOHLC.json', orient="values")
df.to_csv('/home/ubuntu/Desktop/GBP.csv')
graph = pd.read_csv('/home/ubuntu/Desktop/GBP.csv')
graph = graph[['Date','Close']]
graph.set_index('Date', inplace=True, drop=True) 
graph.plot(grid = True)
stock_return = graph.apply(lambda x: x / x[0])
stock_return.head()
stock_return.plot(grid = True).axhline(y = 1, color = "black", lw = 2)
stock_change = graph.apply(lambda x: np.log(x) - np.log(x.shift(1))) # shift moves dates back by 1.
stock_change.head()
stock_change.plot(grid = True).axhline(y = 0, color = "black", lw = 2)
graph["20d"] = np.round(graph["Close"].rolling(window = 20, center = False).mean(), 2)
df = pd.DataFrame()
df['ds'] = stock_return.index
df['y']=graph['Close'].apply(lambda x: np.log(x)).values
df.tail()
m0 = Prophet(yearly_seasonality=True)
#m0 = Prophet(daily_seasonality=False)
m0.fit(df)
#n_add = 365 - len()
n_add = 100
future = m0.make_future_dataframe(periods=n_add, freq='D') 
future.tail(10)
forecast = m0.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()
forcast = m0.plot(forecast, ylabel='$\ln($stock_return$)$');
trend = m0.plot_components(forecast);
forecast['exp_yhat'] = np.exp(forecast['yhat'])
forecast['exp_yhat_lower'] = np.exp(forecast['yhat_lower'])
forecast['exp_yhat_upper'] = np.exp(forecast['yhat_upper'])
col_list = ['ds','exp_yhat', 'exp_yhat_lower', 'exp_yhat_upper']
df = forecast[col_list]
unix = df.ds.apply(lambda x : (x-datetime.datetime(1970,1,1)).total_seconds())
df.drop('ds', axis=1, inplace=True)
df = df.join(unix)
df = df.round(2)
df.tail()
df.to_json(r'/home/ubuntu/Desktop/GBP.json', orient="values")



start = datetime.datetime(2016,6,1)
end = datetime.date.today()
df = data.DataReader("CHFUSD=X", "yahoo", start, end)
df['unix'] = df.index
df['unix'] = df.unix.apply(lambda x : (x-datetime.datetime(1970,1,1)).total_seconds())
df.drop(df.columns.difference(['Date','Open', 'High', 'Low', 'Close', 'unix']), 1, inplace=True)
df = df.round(4)
df.to_json(r'/home/ubuntu/Desktop/CHFOHLC.json', orient="values")
df.to_csv('/home/ubuntu/Desktop/CHF.csv')
graph = pd.read_csv('/home/ubuntu/Desktop/CHF.csv')
graph = graph[['Date','Close']]
graph.set_index('Date', inplace=True, drop=True) 
graph.plot(grid = True)
stock_return = graph.apply(lambda x: x / x[0])
stock_return.head()
stock_return.plot(grid = True).axhline(y = 1, color = "black", lw = 2)
stock_change = graph.apply(lambda x: np.log(x) - np.log(x.shift(1))) # shift moves dates back by 1.
stock_change.head()
stock_change.plot(grid = True).axhline(y = 0, color = "black", lw = 2)
graph["20d"] = np.round(graph["Close"].rolling(window = 20, center = False).mean(), 2)
df = pd.DataFrame()
df['ds'] = stock_return.index
df['y']=graph['Close'].apply(lambda x: np.log(x)).values
df.tail()
m0 = Prophet(yearly_seasonality=True)
#m0 = Prophet(daily_seasonality=False)
m0.fit(df)
#n_add = 365 - len()
n_add = 100
future = m0.make_future_dataframe(periods=n_add, freq='D') 
future.tail(10)
forecast = m0.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()
forcast = m0.plot(forecast, ylabel='$\ln($stock_return$)$');
trend = m0.plot_components(forecast);
forecast['exp_yhat'] = np.exp(forecast['yhat'])
forecast['exp_yhat_lower'] = np.exp(forecast['yhat_lower'])
forecast['exp_yhat_upper'] = np.exp(forecast['yhat_upper'])
col_list = ['ds','exp_yhat', 'exp_yhat_lower', 'exp_yhat_upper']
df = forecast[col_list]
unix = df.ds.apply(lambda x : (x-datetime.datetime(1970,1,1)).total_seconds())
df.drop('ds', axis=1, inplace=True)
df = df.join(unix)
df = df.round(2)
df.tail()
df.to_json(r'/home/ubuntu/Desktop/CHF.json', orient="values")




start = datetime.datetime(2016,6,1)
end = datetime.date.today()
df = data.DataReader("AUD=X", "yahoo", start, end)
df['unix'] = df.index
df['unix'] = df.unix.apply(lambda x : (x-datetime.datetime(1970,1,1)).total_seconds())
df.drop(df.columns.difference(['Date','Open', 'High', 'Low', 'Close', 'unix']), 1, inplace=True)
df = df.round(4)
df.to_json(r'/home/ubuntu/Desktop/AUDOHLC.json', orient="values")
df.to_csv('/home/ubuntu/Desktop/AUD.csv')
graph = pd.read_csv('/home/ubuntu/Desktop/AUD.csv')
graph = graph[['Date','Close']]
graph.set_index('Date', inplace=True, drop=True) 
graph.plot(grid = True)
stock_return = graph.apply(lambda x: x / x[0])
stock_return.head()
stock_return.plot(grid = True).axhline(y = 1, color = "black", lw = 2)
stock_change = graph.apply(lambda x: np.log(x) - np.log(x.shift(1))) # shift moves dates back by 1.
stock_change.head()
stock_change.plot(grid = True).axhline(y = 0, color = "black", lw = 2)
graph["20d"] = np.round(graph["Close"].rolling(window = 20, center = False).mean(), 2)
df = pd.DataFrame()
df['ds'] = stock_return.index
df['y']=graph['Close'].apply(lambda x: np.log(x)).values
df.tail()
m0 = Prophet(yearly_seasonality=True)
#m0 = Prophet(daily_seasonality=False)
m0.fit(df)
#n_add = 365 - len()
n_add = 100
future = m0.make_future_dataframe(periods=n_add, freq='D') 
future.tail(10)
forecast = m0.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()
forcast = m0.plot(forecast, ylabel='$\ln($stock_return$)$');
trend = m0.plot_components(forecast);
forecast['exp_yhat'] = np.exp(forecast['yhat'])
forecast['exp_yhat_lower'] = np.exp(forecast['yhat_lower'])
forecast['exp_yhat_upper'] = np.exp(forecast['yhat_upper'])
col_list = ['ds','exp_yhat', 'exp_yhat_lower', 'exp_yhat_upper']
df = forecast[col_list]
unix = df.ds.apply(lambda x : (x-datetime.datetime(1970,1,1)).total_seconds())
df.drop('ds', axis=1, inplace=True)
df = df.join(unix)
df = df.round(2)
df.tail()
df.to_json(r'/home/ubuntu/Desktop/AUD.json', orient="values")



start = datetime.datetime(2016,6,1)
end = datetime.date.today()
df = data.DataReader("EURUSD=X", "yahoo", start, end)
df['unix'] = df.index
df['unix'] = df.unix.apply(lambda x : (x-datetime.datetime(1970,1,1)).total_seconds())
df.drop(df.columns.difference(['Date','Open', 'High', 'Low', 'Close', 'unix']), 1, inplace=True)
df = df.round(4)
df.to_json(r'/home/ubuntu/Desktop/EUROHLC.json', orient="values")
df.to_csv('/home/ubuntu/Desktop/EUR.csv')
graph = pd.read_csv('/home/ubuntu/Desktop/EUR.csv')
graph = graph[['Date','Close']]
graph.set_index('Date', inplace=True, drop=True) 
graph.plot(grid = True)
stock_return = graph.apply(lambda x: x / x[0])
stock_return.head()
stock_return.plot(grid = True).axhline(y = 1, color = "black", lw = 2)
stock_change = graph.apply(lambda x: np.log(x) - np.log(x.shift(1))) # shift moves dates back by 1.
stock_change.head()
stock_change.plot(grid = True).axhline(y = 0, color = "black", lw = 2)
graph["20d"] = np.round(graph["Close"].rolling(window = 20, center = False).mean(), 2)
df = pd.DataFrame()
df['ds'] = stock_return.index
df['y']=graph['Close'].apply(lambda x: np.log(x)).values
df.tail()
m0 = Prophet(yearly_seasonality=True)
#m0 = Prophet(daily_seasonality=False)
m0.fit(df)
#n_add = 365 - len()
n_add = 100
future = m0.make_future_dataframe(periods=n_add, freq='D') 
future.tail(10)
forecast = m0.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()
forcast = m0.plot(forecast, ylabel='$\ln($stock_return$)$');
trend = m0.plot_components(forecast);
forecast['exp_yhat'] = np.exp(forecast['yhat'])
forecast['exp_yhat_lower'] = np.exp(forecast['yhat_lower'])
forecast['exp_yhat_upper'] = np.exp(forecast['yhat_upper'])
col_list = ['ds','exp_yhat', 'exp_yhat_lower', 'exp_yhat_upper']
df = forecast[col_list]
unix = df.ds.apply(lambda x : (x-datetime.datetime(1970,1,1)).total_seconds())
df.drop('ds', axis=1, inplace=True)
df = df.join(unix)
df = df.round(2)
df.tail()
df.to_json(r'/home/ubuntu/Desktop/EUR.json', orient="values")



