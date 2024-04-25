# Future-Stock-Price-Prediction

## Overview

This project aims to predict stock prices using neural networks. We leverage historical stock price data obtained from Yahoo Finance to train and evaluate our predictive models. Below is a detailed overview of Yahoo Finance and its features that serve as the primary data source for this project.

## Yahoo Finance Overview

Yahoo Finance is a popular and widely used financial platform that offers a comprehensive range of financial information, tools, and services to investors, traders, and financial professionals. It provides access to real-time stock quotes, interactive charts, financial news, portfolio management tools, and historical stock price data for thousands of publicly traded companies worldwide.

### Key Features of Yahoo Finance:

1. **Real-Time Stock Quotes**: 
   - Yahoo Finance offers real-time stock quotes that allow users to track the latest price movements, trading volume, and other key metrics for individual stocks, indices, and commodities.

2. **Interactive Charts**: 
   - Users can access interactive and customizable charts to visualize historical stock price data, technical indicators, and trading patterns. These charts enable users to perform in-depth technical analysis and make informed trading decisions.

3. **Financial News and Analysis**: 
   - Yahoo Finance provides up-to-date financial news, market commentary, and expert analysis from leading financial journalists, columnists, and contributors. This information helps users stay informed about market trends, economic developments, and company-specific news that may impact stock prices.

4. **Portfolio Management Tools**: 
   - Users can create and manage their investment portfolios using Yahoo Finance's portfolio management tools. These tools allow users to track the performance of their investments, monitor asset allocation, and analyze portfolio risk and return metrics.

5. **Historical Stock Price Data**: 
   - Yahoo Finance offers comprehensive historical stock price data for thousands of publicly traded companies. This data includes daily, weekly, monthly, and yearly stock price information, along with other key metrics such as trading volume, dividends, and stock splits. The historical data provided by Yahoo Finance is widely used for financial analysis, forecasting, and research purposes.

6. **Company Profiles and Financial Statements**: 
   - Yahoo Finance provides detailed company profiles, including key executives, business description, industry classification, and financial statements (e.g., income statement, balance sheet, cash flow statement). These resources enable users to conduct fundamental analysis and evaluate the financial health and performance of individual companies.

7. **Screeners and Filters**: 
   - Users can utilize Yahoo Finance's stock screeners and filters to identify investment opportunities based on specific criteria, such as industry sector, market capitalization, valuation metrics (e.g., P/E ratio, P/B ratio), and technical indicators. These tools help users narrow down their investment options and focus on stocks that meet their investment objectives and risk tolerance.

8. **Educational Content and Webinars**: 
   - Yahoo Finance offers educational content, tutorials, and webinars on various topics related to investing, trading, financial planning, and personal finance. These resources are designed to help users improve their financial literacy, learn new investment strategies, and make better-informed financial decisions.

### Data Sources and Availability:

- **Yahoo Finance**: 
  - The primary source of the financial data and information provided by Yahoo Finance is directly sourced from exchanges, financial institutions, regulatory filings, and other reliable sources. Yahoo Finance aggregates, processes, and presents this data in a user-friendly and accessible format.

## Usage:

- **Investment Research and Analysis**: 
  - Yahoo Finance serves as a valuable resource for conducting investment research, performing financial analysis, and evaluating investment opportunities across different asset classes and markets.

- **Market Monitoring and Tracking**: 
  - Users can monitor and track market trends, stock performance, and economic indicators using Yahoo Finance's real-time data and interactive tools.

- **Financial Planning and Decision-making**: 
  - Yahoo Finance helps users make informed financial planning and investment decisions by providing access to comprehensive financial data, analysis tools, and educational resources.
-----
## Dataset Preview

![Dataset Preview](https://github.com/PrabhuTeja19/Future-Stock-Price-Prediction/raw/main/DATASET.png)

## Data Dictionary

The dataset contains the following columns/features:
- **Date**: 
  - The date on which the stock price data was recorded.
- **Open**: 
  - The opening price of the stock on a given trading day.
- **High**: 
  - The highest price at which the stock traded during the trading day.
- **Low**: 
  - The lowest price at which the stock traded during the trading day.
- **Close**: 
  - The closing price of the stock on a given trading day.
- **Adjusted Close**: 
  - The adjusted closing price accounts for factors such as dividends, stock splits, and other corporate actions that affect the stock price.
- **Volume**: 
  - The total number of shares traded during the trading day.
  -------

## Stock Analysis with Candlestick and Seasonal Decomposition

To visualize historical stock price data using candlestick charts and perform seasonal decomposition to identify underlying trends, seasonal patterns, and residuals in the stock prices.

### Overview

The project utilizes Python libraries such as `yfinance`, `pandas`, `plotly`, and `statsmodels` to download, visualize, and analyze historical stock price data for a specified stock symbol.

### Features

- **Download Historical Data**: Downloads historical stock price data using the `yfinance` library.
- **Plot Candlestick Chart**: Visualizes the open, high, low, and adjusted close prices of the stock using `plotly`.
- **Perform Seasonal Decomposition**: Identifies underlying trends, seasonal patterns, and residuals in the stock prices using `seasonal_decompose` from `statsmodels`.

### Packages Used

- `yfinance`: For downloading historical stock price data.
- `matplotlib.pyplot`: For plotting candlestick charts.
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical computations.
- `plotly.graph_objects`: For interactive plotting.
- `statsmodels.tsa.seasonal`: For seasonal decomposition.
- `sklearn.preprocessing.MinMaxScaler`: For data preprocessing.
- `sklearn.model_selection.train_test_split`: For splitting data into training and testing sets.
- `sklearn.metrics`: For evaluating model performance.
- `datetime`: For date and time operations.
- `keras.models.Sequential`: For defining the LSTM model architecture.
- `keras.layers.LSTM`: For LSTM layers in the neural network.
- `keras.metrics.MeanAbsolutePercentageError`: For evaluating the LSTM model performance.

### Execution Order and Graph Interpretation

#### Code Execution Order:

```python
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from plotly.subplots import make_subplots
from datetime import datetime

class StockAnalysis:
    def __init__(self, stock, start_date, end_date):
        self.stock = stock
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    def download_data(self):
        self.data = yf.download(self.stock, self.start_date, self.end_date)
        return self.data

    def plot_candlestick(self, data):
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Adj Close']
        )])
        fig.update_layout(
            title_text=self.stock + " Stock Price",
            xaxis_rangeslider_visible=False,
            yaxis_fixedrange=False,
            yaxis_type="linear"
        )
        fig.show()

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

        fig.update_layout(title_text="Seasonal Decomposition of " + self.stock + " Stock Price",
                          height=800, width=1400,
                          showlegend=False)

        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Trend", row=2, col=1)
        fig.update_yaxes(title_text="Seasonal", row=3, col=1)
        fig.update_yaxes(title_text="Residual", row=4, col=1)

        fig.show()
```

#### Define stock symbol, start date, and end date
stock = 'META'
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - pd.DateOffset(years=10)).strftime('%Y-%m-%d')

#### Create an instance of StockAnalysis class
stock_analysis = StockAnalysis(stock, start_date, end_date)

#### Download data and visualize
Plot_graph = stock_analysis.visualize_data()
data = stock_analysis.download_data()
### Graph Interpretation:

#### Candlestick Chart (Metastock):

![Metastock Candlestick Chart](https://github.com/PrabhuTeja19/Future-Stock-Price-Prediction/blob/main/METASTOCK)

- **Yearwise Candleplot Data**: 
  - The candlestick chart displays the open, high, low, and adjusted close prices of the 'META' stock for each year from 2016 to 2024.
  - **Trend Analysis**: 
    - There is a noticeable increase in the stock prices from 2016 to 2024, indicating a positive trend in the stock performance over the years.
  
#### Seasonal Decomposition:

![Seasonal Decomposition](https://github.com/PrabhuTeja19/Future-Stock-Price-Prediction/blob/main/Decomposition%20Metastock)

- **Original Data**: 
  - Represents the original adjusted close prices of the stock.
- **Trend**: 
  - Represents the long-term trend or pattern in the stock prices.
- **Seasonal**: 
  - Represents the seasonal or periodic fluctuations in the stock prices.
- **Residual**: 
  - Represents the random or irregular fluctuations in the stock prices after removing the trend and seasonal components.




### Data PreProcessing

In the following code snippet, we perform the data preprocessing steps required for preparing the dataset for the stock price prediction model:

1. We consider the first 100 elements in `X`.
2. The next element (101st) is assigned to `y` as the response variable.
3. For the next iteration:
   - We append elements from index 1 to 101 into `X`.
   - We append the 102nd element into `y`.
   
```python
sequence = 100
X = np.array([scale_input[i-sequence:i] for i in range(sequence, len(scale_input))])
y = scale_input[sequence:]
X.shape, y.shape
```
### Data Splitting

In the following code snippet, we split the preprocessed data (`X` and `y`) into training and testing sets using the `train_test_split` function:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
```
```output
X_train shape: (1933, 100, 1)
X_test shape:  (484, 100, 1)
y_train shape: (1933, 1)
y_test shape:  (484, 1)
```

### Model Architecture

In the following code snippet, we define the architecture of the neural network model using the `Sequential` API from Keras:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

Model = Sequential()
Model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
Model.add(LSTM(64, return_sequences=False))
Model.add(Dense(25))
Model.add(Dense(1))
```
The neural network model architecture for stock price prediction is defined as follows:

- **First LSTM Layer**:
  - Number of units: `128`
  - `return_sequences=True`: Returns the full sequence for subsequent layers.
  - `input_shape=(X_train.shape[1], 1)`: Input dimensions matching the shape of the training data.

- **Second LSTM Layer**:
  - Number of units: `64`
  - `return_sequences=False`: Last LSTM layer in the stack, does not return sequences.

- **Dense Layers**:
  - First Dense Layer:
    - Number of units: `25`
  - Second Dense Layer (Output Layer):
    - Number of units: `1`
    - Produces the final output for stock price prediction.
____
### Model Compilation and Training:

In the following code snippet, we compile and train the neural network model for stock price prediction using the `adam` optimizer and `mean_squared_error` loss function:

```python
from tensorflow.keras.losses import MeanAbsolutePercentageError

mape_metric = MeanAbsolutePercentageError()
Model.compile(optimizer="adam", loss="mean_squared_error", metrics=[mape_metric])
Model.fit(X_train, y_train, batch_size=32, epochs=15, validation_data=(X_test, y_test))
```
The neural network model for stock price prediction is compiled and trained with the following configuration:
- **Optimizer**: 
  - `adam` optimizer is used for optimizing the model parameters during training.
- **Loss Function**: 
  - `mean_squared_error` loss function is used to measure the difference between the predicted and actual stock prices.
- **Metrics**: 
  - `MeanAbsolutePercentageError` (MAPE) is used as an additional metric to evaluate the model performance.
- **Batch Size**: 
  - `32` samples per gradient update during training.
- **Epochs**: 
  - `15` epochs are used for training the model.
- **Validation Data**: 
  - The validation data `(X_test, y_test)` is used to evaluate the model performance during training.
----

## StockPredictor Class for Predicting Next 14 Days' Stock Prices

```python
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


stock_predictor = StockPredictor(Model, X_test, y_test)
predictions = stock_predictor.next_14_close(14)
start_date = '2024-04-26'
predictions_df = stock_predictor.generate_predictions_dataframe(predictions, start_date)
predictions_df
```

### Class `StockPredictor`:

This class is designed to predict the next 14 days' closing stock prices using a trained model and the last available data points from the test set.

#### Methods:

1. **`__init__(self, model, X_test, y_test)`**:
   - Initializes the `StockPredictor` class with a trained model (`model`), test input data (`X_test`), and test output data (`y_test`).
2. **`next_14_close(self, days)`**:
   - Predicts the next 14 days' closing stock prices.
   - `days`: Number of days for which predictions are to be made.
   - `predictions`: A list to store the predicted closing prices.
   - `actual`: Takes the last data point from the test set as the initial data point for prediction.
   - `X_pred`: Predicts the next day's closing price using the model.
   - `np.roll(actual, shift=-1, axis=0)`: Shifts the `actual` array to remove the first data point and makes space for the predicted value.
   - Returns: A list of predicted closing prices for the next 14 days.
3. **`generate_predictions_dataframe(self, predictions, start_date)`**:
   - Generates a DataFrame with the predicted closing prices.
   - `predictions`: List of predicted closing prices.
   - `start_date`: The start date for the predictions.
   - `a`: Inverses scales the predicted values to their original scale using the `inverse_transform` method of the scaler (`scale`).
   - `future_dates`: Generates a list of future dates for the next 14 days.
   - `df`: Creates a DataFrame with the predicted closing prices and corresponding future dates.
   - Returns: A DataFrame (`predictions_df`) with the predicted closing prices and dates as indices.

#### Usage:

1. **Initialization**:
   - `stock_predictor = StockPredictor(Model, X_test, y_test)`: Creates an instance of the `StockPredictor` class with the trained model, test input data, and test output data.
2. **Predict Next 14 Days**:
   - `predictions = stock_predictor.next_14_close(14)`: Calls the `next_14_close` method to predict the next 14 days' closing prices.
3. **Generate Predictions DataFrame**:
   - `start_date = '2024-04-26'`: Specifies the start date for the predictions.
   - `predictions_df = stock_predictor.generate_predictions_dataframe(predictions, start_date)`: Calls the `generate_predictions_dataframe` method to create a DataFrame (`predictions_df`) with the predicted closing prices and future dates.

The `predictions_df` DataFrame will contain the predicted closing prices for the next 14 days, starting from the specified `start_date`.

```plaintext
Date        Adj Close
2024-04-26	484.949829
2024-04-29	482.075592
2024-04-30	479.393829
2024-05-01	476.833557
2024-05-02	474.357788
2024-05-03	471.947540
2024-05-06	469.593719
2024-05-07	467.291626
2024-05-08	465.039307
2024-05-09	462.835449
2024-05-10	460.679504
2024-05-13	458.570770
2024-05-14	456.508575
2024-05-15	454.492096
```



      







