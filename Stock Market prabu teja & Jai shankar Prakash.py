# %%
import yfinance as yf
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime,timedelta

# %%
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.metrics import MeanAbsolutePercentageError

# %% [markdown]
# # STOCK ANALYSIS
# 

# %%
class StockAnalysis:
    def __init__(self, stock, start_date, end_date):
        self.stock = stock
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
# We are Downlording the data from Yahoo finance 
    def download_data(self):
        self.data = yf.download(self.stock, self.start_date, self.end_date)
        return self.data
# Using Plotly we are ploting Candle stick graph
    def plot_candlestick(self, data):
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Adj Close']
        )])
        fig.update_layout(
            title_text= self.stock + " Stock Price",
            xaxis_rangeslider_visible=False,
            yaxis_fixedrange=False,
            yaxis_type="linear"
        )
        fig.show()

# To verify the trend and seasonality we have plot seasonal decomposition 
    def visualize_data(self):
        if self.data is None:
            self.data = self.download_data()

        self.plot_candlestick(self.data)

        result = seasonal_decompose(self.data["Adj Close"], model="additive", period=30)

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=("Original Data", "Trend", "Seasonal", "Residual"))

        fig.add_trace(go.Scatter(x=self.data.index, y=self.data["Adj Close"], mode='lines', name="Original Data", line=dict(color='blue')), row=1, col=1)

        fig.add_trace(go.Scatter(x=self.data.index, y=result.trend, mode='lines', name="Trend", line=dict(color='green')), row=2, col=1)

        fig.add_trace(go.Scatter(x=self.data.index, y=result.seasonal, mode='lines', name="Seasonal", line=dict(color='orange')), row=3, col=1)

        fig.add_trace(go.Scatter(x=self.data.index, y=result.resid, mode='lines', name="Residual", line=dict(color='red')), row=4, col=1)

        fig.update_layout(title_text="Seasonal Decomposition of " + self. stock +" Stock Price",
                          height=800, width=1600,
                          showlegend=False)

        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Trend", row=2, col=1)
        fig.update_yaxes(title_text="Seasonal", row=3, col=1)
        fig.update_yaxes(title_text="Residual", row=4, col=1)
        fig.show()



# %%
stock = "META"
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - pd.DateOffset(years=10)).strftime('%Y-%m-%d')
stock_analysis = StockAnalysis(stock, start_date, end_date)
Plot_graph = stock_analysis.visualize_data()
data = stock_analysis.download_data()
data

# %% [markdown]
# ## Close price forcasting using EMA 
# ### Ft+1=kXt+(1−k)Ft,
# ### Ft+1 = forecast for Next day
# ### Ft = forecast of today  
# ### Xt = close price of today
# ### k = Smoothing Factor

# %%
class EMA_Forecaster:
    def __init__(self, data):
        self.data = data
        self.EMAs = {}
        self.forecasted_values = {}

    def calculate_EMA(self, period):
        ema_key = f'EMA_{period}'
        self.EMAs[ema_key] = self.data["Adj Close"].ewm(span=period).mean()

    def plot_EMA(self, periods):
        fig = go.Figure()
        for period in periods:
            ema_key = f'EMA_{period}'
            fig.add_trace(go.Scatter(x=self.EMAs[ema_key].index, y=self.EMAs[ema_key].values, mode='lines', name=f'EMA {period}'))
        fig.update_layout(title='EMA for Different Periods', xaxis_title='Date', yaxis_title='EMA')
        fig.show()

    def forecast_adj_close(self, periods, k, days):
        for period in periods:
            ema_key = f'EMA_{period}'
            EMA_values_forecasted = []
            current_EMA = self.EMAs[ema_key].values[-1]

            for i in range(days):
                a = k * self.data["Adj Close"].values[-1] + (1 - k) * current_EMA
                EMA_values_forecasted.append(a)
                current_EMA = a

            self.forecasted_values[period] = np.array(EMA_values_forecasted)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data["Adj Close"], mode='lines', name='Actual Adj Close'))
            fig.add_trace(go.Scatter(x=pd.date_range(start=self.data.index[-1], periods=days, freq='B'), y=EMA_values_forecasted, mode='lines', name=f'Forecasted Adj Close - EMA {period}'))
            fig.update_layout(title=f'Forecasted Adj Close for Next {days} Days - EMA {period}', xaxis_title='Date', yaxis_title='Adjusted Close Price')
            fig.show()

            # Calculate accuracy metrics
            actual = self.data["Adj Close"].values[-days:]
            mae = mean_absolute_error(actual, EMA_values_forecasted)
            rmse = np.sqrt(mean_squared_error(actual, EMA_values_forecasted))
            mape = np.mean(np.abs((actual - EMA_values_forecasted) / actual)) * 100
            print(f"EMA {period} Forecast Accuracy (MAE): {mae:.2f}")
            print(f"EMA {period} Forecast Accuracy (RMSE): {rmse:.2f}")
            print(f"EMA {period} Forecast Accuracy (MAPE): {mape:.2f}%")

        return self.forecasted_values

    def print_forecasted_values_as_dataframe(self):
        df = pd.DataFrame(self.forecasted_values)
        df.index = pd.date_range(start=self.data.index[-1], periods=len(df), freq='B')
        print("Forecasted Adjusted Close Values:")
        print(df)

# Assuming 'data' is your DataFrame containing historical stock data
ema_forecaster = EMA_Forecaster(data)

ema_forecaster.calculate_EMA(50)
ema_forecaster.calculate_EMA(100)
ema_forecaster.calculate_EMA(150)
ema_forecaster.calculate_EMA(200)
ema_forecaster.calculate_EMA(250)

forecasted_values = ema_forecaster.forecast_adj_close([50, 100, 150, 200, 250], 0.5, 14)

ema_forecaster.print_forecasted_values_as_dataframe()

# %% [markdown]
# ### SCALING THE DATA

# %%
input = data[["Adj Close"]]
input

# %%
scale = MinMaxScaler(feature_range= (0,1))
scale_input = scale.fit_transform(input)
scale_input

# %% [markdown]
# ### Data PreProcessing
# 
# In this below code we are considering the first 100 element in X
# And next 101 is going into y = response
# for next iteration 2 -101 append into X
# in y we will append 102 element

# %%
sequence = 100
X = np.array([scale_input[i-sequence:i] for i in range(sequence, len(scale_input))])
y = scale_input[sequence:]
X.shape, y.shape

# %% [markdown]
# ### Train Test Split

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train.shape,X_test.shape,y_train.shape,y_test.shape

# %% [markdown]
# ### Implementing Neural Network

# %%
Model = Sequential()
Model.add(LSTM(128,return_sequences= True,input_shape = (X_train.shape[1],1)))
Model.add(LSTM(64,return_sequences= False))
Model.add(Dense(25))
Model.add(Dense(1))

# %%
mape_metric = MeanAbsolutePercentageError()
Model.compile(optimizer="adam", loss="mean_squared_error",metrics=[mape_metric])

# %%
Model = Model.fit(X_train,y_train,batch_size= 32,epochs= 15,validation_data=(X_test, y_test))
Model

# %% [markdown]
# ### Checking the model loss accuracy using plot

# %%
train_loss = Model.history['loss']
val_loss = Model.history['val_loss']
epochs = range(1, len(train_loss) + 1)

fig = go.Figure()
fig.add_trace(go.Scatter(x=list(epochs), y=train_loss, mode='lines+markers', name='Training Loss', marker=dict(symbol='circle')))
fig.add_trace(go.Scatter(x=list(epochs), y=val_loss, mode='lines+markers', name='Validation Loss', marker=dict(symbol='x')))

fig.update_layout(title='Training and Validation Loss',
                  xaxis_title='Epoch',
                  yaxis_title='Loss')

fig.show()


# %% [markdown]
# ### Predicting the Next 14 days stock Adj price using LSTM

# %%

class StockPredictor:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def next_14_close(self, days):
        predictions = []
        actual = self.X_test[-1]
        for i in range(days):
            X_pred = self.model.predict(np.expand_dims(actual, axis=0))
            predictions.append(X_pred[0][0])
            print(predictions)
            actual = np.roll(actual, shift=-1, axis=0)
            actual[-1] = X_pred[0][0]
        return predictions

    def generate_predictions_dataframe(self, predictions, start_date):
        a = scale.inverse_transform(np.array(predictions).reshape(-1, 1))
        future_dates = pd.date_range(start=start_date, periods=len(predictions), freq='B').tolist()
        df = pd.DataFrame({'Adj Close': a.reshape(-1)}, index=future_dates)
        return df


stock_predictor = StockPredictor(Model.model, X_test, y_test)
predictions = stock_predictor.next_14_close(20)
start_date = '2024-04-23'
predictions_df = stock_predictor.generate_predictions_dataframe(predictions, start_date)

# Combine actual and forecasted data
combined_data = pd.concat([data[['Adj Close']], predictions_df], axis=0)
combined_data

# Plot actual and forecasted prices
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Adj Close'], mode='lines', name='Actual Price', line=dict(color='light blue')))
fig.add_trace(go.Scatter(x=predictions_df.index, y=predictions_df['Adj Close'], mode='lines', name='Forecasted Price', line=dict(color='green')))
fig.update_layout(title='Actual vs Forecasted Prices', xaxis_title='Date', yaxis_title='Adjusted Close Price',
                  xaxis=dict(type='date', tickformat='%Y-%m-%d', range=[data.index[0], predictions_df.index[-1]]))
fig.show()



