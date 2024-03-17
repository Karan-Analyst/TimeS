import streamlit as st
import yfinance as yf
from prophet import Prophet
from datetime import date, timedelta
# from prophet.plot_plotly import plot_plotly,plot_components
from plotly import graph_objs as go
from prophet.plot import plot_plotly
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
from sklearn.model_selection import train_test_split

st.title("Stock Performance")
stocks = (
    'UPL',
    'BHARTIARTL',
    'HDFCLIFE',
    'BAJFINANCE',
    'ADANIPORTS',
    'ADANIENT',
    'HINDALCO',
    'TATACONSUM',
    'BRITANNIA',
    'TCS',
    'INDUSINDBK',
    'POWERGRID',
    'HDFCBANK',
    'SBILIFE',
    'TITAN',
    'BAJAJFINSV',
    'MARUTI',
    'DRREDDY',
    'JSWSTEEL',
    'HINDUNILVR',
    'APOLLOHOSP',
    'BAJAJ-AUTO',
    'TATASTEEL',
    'KOTAKBANK',
    'NESTLEIND',
    'WIPRO',
    'ITC',
    'EICHERMOT',
    'RELIANCE',
    'ICICIBANK',
    'CIPLA',
    'AXISBANK',
    'ULTRACEMCO',
    'SUNPHARMA',
    'DIVISLAB',
    'SBIN',
    'ASIANPAINT',
    'TECHM',
    'GRASIM',
    'INFY',
    'LTIM',
    'ONGC',
    'HCLTECH',
    'LT',
    'NTPC',
    'HEROMOTOCO',
    'TATAMOTORS',
    'COALINDIA',
    'BPCL',
    'M&M')
# Add '.NS' to each string item
yahoo_list = tuple(item + ".NS" for item in stocks)

selected_stocks = st.selectbox("Select Stock", yahoo_list)
Start = (date.today() - timedelta(days=365 * 10)).strftime("%Y-%m-%d")
Today = date.today().strftime("%Y-%m-%d")


@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, Start, Today)
    data.reset_index(inplace=True)
    data['200_avg'] = data['Close'].rolling(window=200).mean()
    data.dropna(inplace=True)
    return data


data_load_state = st.text("Load data...")
data = load_data(selected_stocks)
data_load_state.text("Loading data.. ok!")

st.subheader('Raw Data')
st.write(data.tail())


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text=selected_stocks + " 10 year data with slider at bottom",
                      xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()

# Forecasting
n_years = st.slider("No of Years: ", 1, 5)
period = n_years * 365

df_train = data[['Date', 'Close', '200_avg']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
# Split the data into training and testing sets
train_data, test_data = train_test_split(df_train, test_size=0.15, random_state=42)

# Print the shapes of the training and testing sets
print("Training set shape:", train_data.shape)
print("Testing set shape:", test_data.shape)


m = Prophet()
#m.add_regressor('200_avg')
m.fit(train_data)

tested_df = m.predict(test_data)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast Data')

st.write('Forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('Forecast Components')
st.write(forecast['weekly'])
fig2 = m.plot_components(forecast)

st.write(fig2)

# a = yf.Ticker(selected_stocks).info['beta']
# st.write(a)

forecasted_values = tested_df['yhat'].values
actual_values = test_data['y'].values

# Compute MAE and RMSE
mae = mean_absolute_error(actual_values, forecasted_values)
rmse = np.sqrt(mean_squared_error(actual_values, forecasted_values))
mape = mean_absolute_percentage_error(actual_values, forecasted_values)

st.write("Mean Absolute Error (MAE):", mae)
st.write("Root Mean Squared Error (RMSE):", rmse)
st.write("Mean Absolute Percentage Error (MAPE):", mape)
