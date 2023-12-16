# DogeCoin Price Prediction with LSTM

**Dataset:** https://www.kaggle.com/datasets/neelgajare/dogecoin-historical-price-data


**Jupyter Notebook:** https://github.com/qckhxnh/Doge-Coin-Price-Prediction


**Acknowledgment:** I have used the help of ChatGPT to learn about the LSTM model and how to implement it in Python. However, I have not copied any text or code directly from this channel. I have also mentioned OpenAI in the citations list.



## I. Introduction

DogeCoin (DOGE) is a cryptocurrency created as a humorous response to the speculative nature of cryptocurrencies. Initially considered a joke, it gained substantial popularity and reached a market capitalization of over $85 billion in May 2021. This project utilizes the DogeCoin Historical Price Dataset to perform time series prediction of DOGE prices using machine learning.

### Dataset

The dataset consists of 1018 text files, each containing historical data for DogeCoin. The columns include:

- Date
- Open
- High
- Low
- Close
- Adjusted close price
- Volume

## II. Machine Learning Model

The machine learning model employed is a Long Short-Term Memory (LSTM) neural network, a type of recurrent neural network (RNN) designed for sequential data. The model is trained on historical Dogecoin price data to predict future prices.

### Building the LSTM Model

```python
# Function to create sequences for training
def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 20
X, y = create_sequences(df_normalized, look_back)

# Building the LSTM model
model = Sequential()
model.add(LSTM(units=50, input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
model.fit(X_train, y_train, epochs=50, batch_size=64)
```

### III. Results

#### Making Predictions

```python
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
```

#### Model Evaluation: Mean Squared Error

```python
mse = mean_squared_error(df['Close'].iloc[train_size + look_back:], predictions)
print(f'Mean Squared Error: {mse}')
```

Mean Squared Error: 0.0006048302769427443

#### Plotting the Results

```python
plt.figure(figsize=(12, 6))
plt.plot(df.index[train_size + look_back:], df['Close'].iloc[train_size + look_back:], label='Actual Price')
plt.plot(df.index[train_size + look_back:], predictions, label='Predicted Price', color='red')
plt.title('Doge-Coin Price Prediction with LSTM')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()
```

![image](https://github.com/qckhxnh/Doge-Coin-Price-Prediction/assets/117861644/4e552174-aaf9-4bfc-9c41-0efeaa0c8ccd)

## IV. References

- Time Series forecasting and Recurrent neural network. [TensorFlow Core](https://www.tensorflow.org/tutorials/structured_data/time_series#recurrent_neural_network).
- OpenAI. (2023). GPT-3.5, a product of OpenAI. [OpenAI ChatGPT](https://chat.openai.com).
