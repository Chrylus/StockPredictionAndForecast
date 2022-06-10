import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from sqlalchemy import false
import streamlit as st
from datetime import date, timedelta
import yfinance as yf
from sklearn import metrics
import plotly.graph_objs as go
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor
from plotly.subplots import make_subplots

today = date.today()
last10 = today - timedelta(10 * 365)
start = last10
end = today

st.title('Stock Trend Prediction')
user_input = st.text_input('Enter stock ticker from yahoo finance (https://finance.yahoo.com/)', 'AAPL')
df = data.DataReader(user_input, 'yahoo', start, end)
df = df.reset_index()

# get company name
tick = yf.Ticker(user_input)

company = tick.info['longName']

st.text("You are viewing '" + company + "' stock price")

# decribing data
st.subheader("Data from " + str(start.year) + " - " + str(end.year))
st.write(df)
st.write(df.describe())

# Data Analysis
st.subheader("Data Analysis")
figx = make_subplots(rows=4, cols=1,subplot_titles=('Open','High','Low','Close'))
figx.add_trace(
    go.Line(x=df.Date, y=df.Open),
    row=1, col=1,
)

figx.add_trace(
    go.Line(x=df.Date, y=df.High),
    row=2, col=1
)

figx.add_trace(
    go.Line(x=df.Date, y=df.Low),
    row=3, col=1
)

figx.add_trace(
    go.Line(x=df.Date, y=df.Close),
    row=4, col=1
)
figx.update_layout(height=1400, width=1000, title_text="OHLC Line Plots", showlegend=False)
st.plotly_chart(figx, use_container_width=True)

st.write("Finding pattern on close price based on seasonal, because we are going to predict the close price")
decomposition = seasonal_decompose(df.Close.head(len(df)), model='additive', period = 30)
fig, axes = plt.subplots(4, 1, sharex=True)
decomposition.observed.plot(ax=axes[0], legend=False, color='r')
axes[0].set_title('Close', color="white")
axes[0].set_ylabel('Observed')
axes[0].yaxis.label.set_color('white')
decomposition.trend.plot(ax=axes[1], legend=False, color='g')
axes[1].set_ylabel('Trend')
axes[1].yaxis.label.set_color('white')
decomposition.seasonal.plot(ax=axes[2], legend=False)
axes[2].set_ylabel('Seasonal')
axes[2].yaxis.label.set_color('white')
decomposition.resid.plot(ax=axes[3], legend=False, color='k')
axes[3].set_ylabel('Residual')
axes[3].yaxis.label.set_color('white')
st.plotly_chart(fig, use_container_width=True)

# Choose number of feature
inputDay = st.slider("Number of input for prediction",50 ,100)
st.write("Input days: " + str(inputDay))

# splitting data intro train and test set
data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)]) 
data_test = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])


# scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_train_array = scaler.fit_transform(data_train)

# splitting independent and dependent variable for training
x_train = []
y_train = []

for i in range (inputDay, data_train_array.shape[0]) :
    x_train.append(data_train_array[i-inputDay : i])
    y_train.append(data_train_array[i, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train_array = x_train.reshape(x_train.shape[0], x_train.shape[1])

# creating model
model = RandomForestRegressor(n_estimators=500, random_state=42, max_depth=10)
# fitting model
model.fit(x_train_array, y_train)

# testing part
past_100_days = data_train.tail(inputDay)

final_df = past_100_days.append(data_test, ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range (inputDay, input_data.shape[0]) :
    x_test.append(input_data[i-inputDay : i])
    y_test.append(input_data[i, 0])
    
x_test, y_test = np.array(x_test), np.array(y_test)
x_test_array = x_test.reshape(x_test.shape[0], x_test.shape[1])
y_predicted = model.predict(x_test_array)
error = round(metrics.mean_absolute_error(y_test, y_predicted) * 100, 2)

scale = scaler.scale_
scale_factor = 1/scale[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# final graph
st.subheader('Prediction vs Original')
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time', fontsize=18)
plt.ylabel('Price', fontsize=18)
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
plt.legend(labelcolor='white')
plt.show()
st.plotly_chart(fig, use_container_width=True)
st.text("error: " + str(error) + "%")

# forecasting
# load model  n preparing data
model_forecast = Prophet()
fr_train = df[['Date', 'Close']]
fr_train = fr_train.rename(columns={'Date': 'ds', 'Close': 'y'})

# select year range
year = st.slider("Prediction year:",1 ,4)
period = year * 365

model_forecast.fit(fr_train)
future = model_forecast.make_future_dataframe(periods=period)
forecast = model_forecast.predict(future)

# plotting forecast
st.subheader('Forecast for next ' + str(year) + " years")
fig1 = plot_plotly(model_forecast, forecast, xlabel='Date', ylabel='Price')
st.plotly_chart(fig1, use_container_width=True)
