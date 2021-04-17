import pandas as pd
from pandas_datareader import data, wb  
import datetime
import numpy as np 
import matplotlib.pyplot as plt
import numpy as np
from fbprophet import Prophet


df = pd.read_json('https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=1000')
df.to_json(r'/home/ubuntu/Desktop/BTC.json', orient="table")
df.to_csv('/home/ubuntu/Desktop/BTC.csv')
graph = pd.read_csv('/home/ubuntu/Desktop/BTC.csv')
graph['Date'] = pd.to_datetime(graph['0'],unit='ms')
graph = graph[['Date','4']]
graph.set_index('Date', inplace=True, drop=True) 
type(graph)
graph.plot(grid = True)
stock_return = graph.apply(lambda x: x / x[0])
stock_return.head()
stock_return.plot(grid = True).axhline(y = 1, color = "black", lw = 2)
stock_change = graph.apply(lambda x: np.log(x) - np.log(x.shift(1))) 
stock_change.head()
stock_change.plot(grid = True).axhline(y = 0, color = "black", lw = 2)
graph["20d"] = np.round(graph["4"].rolling(window = 20, center = False).mean(), 2)
df = pd.DataFrame()
df['ds'] = stock_return.index
df['y']=graph['4'].apply(lambda x: np.log(x)).values
m0 = Prophet(daily_seasonality=False)
m0.fit(df)
n_add = 48
future = m0.make_future_dataframe(periods=n_add, freq='H') 
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
df.to_json(r'/home/ubuntu/Desktop/BTC.json', orient="values")


df = pd.read_json('https://api.binance.com/api/v3/klines?symbol=ETHUSDT&interval=1h&limit=1000')
df.to_json(r'/home/ubuntu/Desktop/ETH.json', orient="table")
df.to_csv('/home/ubuntu/Desktop/ETH.csv')
graph = pd.read_csv('/home/ubuntu/Desktop/ETH.csv')
graph['Date'] = pd.to_datetime(graph['0'],unit='ms')
graph = graph[['Date','4']]
graph.set_index('Date', inplace=True, drop=True) 
type(graph)
graph.plot(grid = True)
stock_return = graph.apply(lambda x: x / x[0])
stock_return.head()
stock_return.plot(grid = True).axhline(y = 1, color = "black", lw = 2)
stock_change = graph.apply(lambda x: np.log(x) - np.log(x.shift(1))) 
stock_change.head()
stock_change.plot(grid = True).axhline(y = 0, color = "black", lw = 2)
graph["20d"] = np.round(graph["4"].rolling(window = 20, center = False).mean(), 2)
df = pd.DataFrame()
df['ds'] = stock_return.index
df['y']=graph['4'].apply(lambda x: np.log(x)).values
m0 = Prophet(daily_seasonality=False)
m0.fit(df)
n_add = 48
future = m0.make_future_dataframe(periods=n_add, freq='H') 
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
df.to_json(r'/home/ubuntu/Desktop/ETH.json', orient="values")


df = pd.read_json('https://api.binance.com/api/v3/klines?symbol=XMRUSDT&interval=1h&limit=1000')
df.to_json(r'/home/ubuntu/Desktop/XMR.json', orient="table")
df.to_csv('/home/ubuntu/Desktop/XMR.csv')
graph = pd.read_csv('/home/ubuntu/Desktop/XMR.csv')
graph['Date'] = pd.to_datetime(graph['0'],unit='ms')
graph = graph[['Date','4']]
graph.set_index('Date', inplace=True, drop=True) 
type(graph)
graph.plot(grid = True)
stock_return = graph.apply(lambda x: x / x[0])
stock_return.head()
stock_return.plot(grid = True).axhline(y = 1, color = "black", lw = 2)
stock_change = graph.apply(lambda x: np.log(x) - np.log(x.shift(1))) 
stock_change.head()
stock_change.plot(grid = True).axhline(y = 0, color = "black", lw = 2)
graph["20d"] = np.round(graph["4"].rolling(window = 20, center = False).mean(), 2)
df = pd.DataFrame()
df['ds'] = stock_return.index
df['y']=graph['4'].apply(lambda x: np.log(x)).values
m0 = Prophet(daily_seasonality=False)
m0.fit(df)
n_add = 48
future = m0.make_future_dataframe(periods=n_add, freq='H') 
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
df.to_json(r'/home/ubuntu/Desktop/XMR.json', orient="values")

df = pd.read_json('https://api.binance.com/api/v3/klines?symbol=BNBUSDT&interval=1h&limit=1000')
df.to_json(r'/home/ubuntu/Desktop/BNB.json', orient="table")
df.to_csv('/home/ubuntu/Desktop/BNB.csv')
graph = pd.read_csv('/home/ubuntu/Desktop/BNB.csv')
graph['Date'] = pd.to_datetime(graph['0'],unit='ms')
graph = graph[['Date','4']]
graph.set_index('Date', inplace=True, drop=True) 
type(graph)
graph.plot(grid = True)
stock_return = graph.apply(lambda x: x / x[0])
stock_return.head()
stock_return.plot(grid = True).axhline(y = 1, color = "black", lw = 2)
stock_change = graph.apply(lambda x: np.log(x) - np.log(x.shift(1))) 
stock_change.head()
stock_change.plot(grid = True).axhline(y = 0, color = "black", lw = 2)
graph["20d"] = np.round(graph["4"].rolling(window = 20, center = False).mean(), 2)
df = pd.DataFrame()
df['ds'] = stock_return.index
df['y']=graph['4'].apply(lambda x: np.log(x)).values
m0 = Prophet(daily_seasonality=False)
m0.fit(df)
n_add = 48
future = m0.make_future_dataframe(periods=n_add, freq='H') 
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
df.to_json(r'/home/ubuntu/Desktop/BNB.json', orient="values")

df = pd.read_json('https://api.binance.com/api/v3/klines?symbol=DOGEUSDT&interval=1h&limit=1000')
df.to_json(r'/home/ubuntu/Desktop/DOGE.json', orient="table")
df.to_csv('/home/ubuntu/Desktop/DOGE.csv')
graph = pd.read_csv('/home/ubuntu/Desktop/DOGE.csv')
graph['Date'] = pd.to_datetime(graph['0'],unit='ms')
graph = graph[['Date','4']]
graph.set_index('Date', inplace=True, drop=True) 
type(graph)
graph.plot(grid = True)
stock_return = graph.apply(lambda x: x / x[0])
stock_return.head()
stock_return.plot(grid = True).axhline(y = 1, color = "black", lw = 2)
stock_change = graph.apply(lambda x: np.log(x) - np.log(x.shift(1))) 
stock_change.head()
stock_change.plot(grid = True).axhline(y = 0, color = "black", lw = 2)
graph["20d"] = np.round(graph["4"].rolling(window = 20, center = False).mean(), 2)
df = pd.DataFrame()
df['ds'] = stock_return.index
df['y']=graph['4'].apply(lambda x: np.log(x)).values
m0 = Prophet(daily_seasonality=False)
m0.fit(df)
n_add = 48
future = m0.make_future_dataframe(periods=n_add, freq='H') 
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
df.to_json(r'/home/ubuntu/Desktop/DOGE.json', orient="values")

df = pd.read_json('https://api.binance.com/api/v3/klines?symbol=XRPUSDT&interval=1h&limit=1000')
df.to_json(r'/home/ubuntu/Desktop/XRP.json', orient="table")
df.to_csv('/home/ubuntu/Desktop/XRP.csv')
graph = pd.read_csv('/home/ubuntu/Desktop/XRP.csv')
graph['Date'] = pd.to_datetime(graph['0'],unit='ms')
graph = graph[['Date','4']]
graph.set_index('Date', inplace=True, drop=True) 
type(graph)
graph.plot(grid = True)
stock_return = graph.apply(lambda x: x / x[0])
stock_return.head()
stock_return.plot(grid = True).axhline(y = 1, color = "black", lw = 2)
stock_change = graph.apply(lambda x: np.log(x) - np.log(x.shift(1))) 
stock_change.head()
stock_change.plot(grid = True).axhline(y = 0, color = "black", lw = 2)
graph["20d"] = np.round(graph["4"].rolling(window = 20, center = False).mean(), 2)
df = pd.DataFrame()
df['ds'] = stock_return.index
df['y']=graph['4'].apply(lambda x: np.log(x)).values
m0 = Prophet(daily_seasonality=False)
m0.fit(df)
n_add = 48
future = m0.make_future_dataframe(periods=n_add, freq='H') 
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
df.to_json(r'/home/ubuntu/Desktop/XRP.json', orient="values")

df = pd.read_json('https://api.binance.com/api/v3/klines?symbol=LTCUSDT&interval=1h&limit=1000')
df.to_json(r'/home/ubuntu/Desktop/LTC.json', orient="table")
df.to_csv('/home/ubuntu/Desktop/LTC.csv')
graph = pd.read_csv('/home/ubuntu/Desktop/LTC.csv')
graph['Date'] = pd.to_datetime(graph['0'],unit='ms')
graph = graph[['Date','4']]
graph.set_index('Date', inplace=True, drop=True) 
type(graph)
graph.plot(grid = True)
stock_return = graph.apply(lambda x: x / x[0])
stock_return.head()
stock_return.plot(grid = True).axhline(y = 1, color = "black", lw = 2)
stock_change = graph.apply(lambda x: np.log(x) - np.log(x.shift(1))) 
stock_change.head()
stock_change.plot(grid = True).axhline(y = 0, color = "black", lw = 2)
graph["20d"] = np.round(graph["4"].rolling(window = 20, center = False).mean(), 2)
df = pd.DataFrame()
df['ds'] = stock_return.index
df['y']=graph['4'].apply(lambda x: np.log(x)).values
m0 = Prophet(daily_seasonality=False)
m0.fit(df)
n_add = 48
future = m0.make_future_dataframe(periods=n_add, freq='H') 
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
df.to_json(r'/home/ubuntu/Desktop/LTC.json', orient="values")

df = pd.read_json('https://api.binance.com/api/v3/klines?symbol=LINKUSDT&interval=1h&limit=1000')
df.to_json(r'/home/ubuntu/Desktop/LINK.json', orient="table")
df.to_csv('/home/ubuntu/Desktop/LINK.csv')
graph = pd.read_csv('/home/ubuntu/Desktop/LINK.csv')
graph['Date'] = pd.to_datetime(graph['0'],unit='ms')
graph = graph[['Date','4']]
graph.set_index('Date', inplace=True, drop=True) 
type(graph)
graph.plot(grid = True)
stock_return = graph.apply(lambda x: x / x[0])
stock_return.head()
stock_return.plot(grid = True).axhline(y = 1, color = "black", lw = 2)
stock_change = graph.apply(lambda x: np.log(x) - np.log(x.shift(1))) 
stock_change.head()
stock_change.plot(grid = True).axhline(y = 0, color = "black", lw = 2)
graph["20d"] = np.round(graph["4"].rolling(window = 20, center = False).mean(), 2)
df = pd.DataFrame()
df['ds'] = stock_return.index
df['y']=graph['4'].apply(lambda x: np.log(x)).values
m0 = Prophet(daily_seasonality=False)
m0.fit(df)
n_add = 48
future = m0.make_future_dataframe(periods=n_add, freq='H') 
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
df.to_json(r'/home/ubuntu/Desktop/LINK.json', orient="values")




