import streamlit as st
import yfinance as yf
from prophet import Prophet
from datetime import date, timedelta
# from prophet.plot_plotly import plot_plotly
from plotly import graph_objs as go
from prophet.plot import plot_plotly
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
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
    return data


data_load_state = st.text("Load data...")
data = load_data(selected_stocks)
data_load_state.text("Loading data.. ok!")

st.subheader('Raw Data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text=selected_stocks+" 10 year data with slider at bottom", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#Forecasting
n_years = st.slider("No of Years: ", 1, 5)
period = n_years * 365
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds","Close":"y"})

m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast Data')
st.write(forecast.tail())

st.write('Forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('Forecast Components')
fig2 = m.plot_components(forecast)

# Remove all components except for yearly
for ax in fig2.get_axes():
    if 'yearly' not in ax.get_title().lower():
        ax.remove()

# Show the modified plot
st.write(fig2)



# Assuming `forecast` contains the forecasted values and `df_test` contains the actual values
forecasted_values = forecast['yhat'].values[-len(df_train):]  # Extract forecasted values for the test period
actual_values = df_train['y'].values  # Replace 'your_actual_column' with the column containing actual values

# Compute MAE and RMSE
mae = mean_absolute_error(actual_values, forecasted_values)
rmse = np.sqrt(mean_squared_error(actual_values, forecasted_values))
mape = mean_absolute_percentage_error(actual_values, forecasted_values)

st.write("Mean Absolute Error (MAE):", mae)
st.write("Root Mean Squared Error (RMSE):", rmse)
st.write("Mean Absolute Percentage Error (MAPE):", mape)
