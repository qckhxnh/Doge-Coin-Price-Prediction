# Doge-Coin-Price-Prediction

Dataset: https://www.kaggle.com/datasets/neelgajare/dogecoin-historical-price-data
Jupyter Notebook: https://github.com/qckhxnh/Doge-Coin-Price-Prediction
Acknowledgment: I have used the help of ChatGPT to learn about the LSTM model and how to implement it in Python. However, I have not copied any text or code directly from this channel. I have also mentioned OpenAI in the citations list.


I. Introduction:
DogeCoin (DOGE) is a cryptocurrency created by two software engineers, Billy Markus and Jackson Palmer, named after the Shiba Inu dog from the "doge" meme. It began as a 'joke' to mock the irrational speculation in cryptocurrencies at the time. Some, however, believe it is a viable investment opportunity. DOGE was introduced on December 6, 2013, and quickly grew its online community, eventually reaching a market capitalization of more than $85 billion on May 5, 2021.

The DogeCoin Historical Price Dataset contains 1018 text files with comma-separated values, each representing historical data for each company. The data in each file contains daily stock data for the company from when it became public to July 28, 2022.
There are 7 columns:
Date
Open
High
Low
Close
Adjusted close price for splits and dividend and capital gain distributions
Volume
We will use this dataset in this project and perform simple machine learning with Python on a Jupyter Notebook to predict the price of DOGE over time. The code is developed in Python and utilizes popular libraries such as pandas for data manipulation, scikit-learn for machine learning, and TensorFlow for deep learning.

II. Machine Learning Model: 
The machine learning model employed in this project is a Long Short-Term Memory (LSTM) neural network. LSTMs are a type of recurrent neural network (RNN) designed to capture and learn patterns in sequential data, making them well-suited for time series prediction tasks. The model is trained on historical Dogecoin price data to learn patterns and trends that can be used to make predictions about future prices.

Building the LSTM Model:
def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)


look_back = 20
X, y = create_sequences(df_normalized, look_back)

The LSTM model is built with the Sequential class, which represents a linear stack of layers. To add layers to the model, use the add method.
The LSTM layer (LSTM(units=50, input_shape=(X_train.shape[1], 1)) is the first to be added. This layer contains 50 neurons and accepts input sequences with the shape (X_train.shape[1], 1). This input shape represents the number of time steps (X_train.shape[1]) and features (1, as we are working with univariate time series data).
The second layer that has been added is fully connected (Dense(units=1)) with a single unit that serves as the output layer for predicting the next value in the sequence.

Compiling the Model:
model = Sequential()
model.add(LSTM(units=50, input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

To prepare the model for training, use the compile method. The Adam optimizer is used, and the loss function is the mean squared error (mean_squared_error). The optimizer and loss function used can have an impact on the model's training dynamics.

Training the Model:
model.fit(X_train, y_train, epochs=50, batch_size=64)

To train the model on the training data (X_train and y_train), the fit method is used. The training procedure entails iterating over the data for a predetermined number of epochs (in this case, 50) and updating the model's weights to minimize the mean squared error.
The batch_size parameter is set to 32, indicating that for each iteration, the training data is divided into batches of 32 samples.

III. Results:
Making predictions:
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

The predict method is used to make predictions on the test data (X_test). The output (predicted_prices) contains the model's predictions for the closing prices of Dogecoin.
Since the data was normalized during training, it's necessary to inverse-transform the predicted prices back to their original scale. This is done using the inverse_transform method of the scaler.

Model evaluation: Mean Squared Error
mse = mean_squared_error(df['Close'].iloc[train_size + look_back:], predictions)
print(f'Mean Squared Error: {mse}')

Mean Squared Error: 0.0006048302769427443
The MSE of 0.0006048302769427443 is relatively low, indicating that the LSTM model predicts Dogecoin prices with a small average squared difference. 

Plotting the results:
plt.figure(figsize=(12, 6))
plt.plot(df.index[train_size + look_back:], df['Close'].iloc[train_size + look_back:], label='Actual Price')
plt.plot(df.index[train_size + look_back:], predictions, label='Predicted Price', color='red')
plt.title('Doge-Coin Price Prediction with LSTM')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()




IV. References:
Time Series forecasting and Recurrent neural network.  Tensorflow Core (no date) TensorFlow. Available at: https://www.tensorflow.org/tutorials/structured_data/time_series#recurrent_neural_network (Accessed: 22 November 2023). 

OpenAI. (2023). GPT-3.5, a product of OpenAI. Accessed on November, 2023. Available at: https://chat.openai.com.
