# Doge-Coin-Price-Prediction

**Dataset:** https://www.kaggle.com/datasets/neelgajare/dogecoin-historical-price-data
**Jupyter Notebook:** https://github.com/qckhxnh/Doge-Coin-Price-Prediction
**Acknowledgment:** I have used the help of ChatGPT to learn about the LSTM model and how to implement it in Python. However, I have not copied any text or code directly from this channel. I have also mentioned OpenAI in the citations list.


**I. Introduction:**

DogeCoin (DOGE) is a cryptocurrency created by two software engineers, Billy Markus and Jackson Palmer, named after the Shiba Inu dog from the "doge" meme. It began as a 'joke' to mock the irrational speculation in cryptocurrencies at the time. Some, however, believe it is a viable investment opportunity. DOGE was introduced on December 6, 2013, and quickly grew its online community, eventually reaching a market capitalization of more than $85 billion on May 5, 2021.

The DogeCoin Historical Price Dataset contains 1018 text files with comma-separated values, each representing historical data for each company. The data in each file contains daily stock data for the company from when it became public to July 28, 2022.
There are 7 columns:

- Date

- Open

- High

- Low

- Close

- Adjusted close price for splits and dividend and capital gain distributions

- Volume

We will use this dataset in this project and perform simple machine learning with Python on a Jupyter Notebook to predict the price of DOGE over time. The code is developed in Python and utilizes popular libraries such as pandas for data manipulation, scikit-learn for machine learning, and TensorFlow for deep learning.

**II. Machine Learning Model: **

The machine learning model employed in this project is a Long Short-Term Memory (LSTM) neural network. LSTMs are a type of recurrent neural network (RNN) designed to capture and learn patterns in sequential data, making them well-suited for time series prediction tasks. The model is trained on historical Dogecoin price data to learn patterns and trends that can be used to make predictions about future prices.

**III. Results:**

The prediction method is used to make predictions on the test data (X_test). The output (predicted_prices) contains the model's predictions for the closing prices of Dogecoin. Since the data was normalized during training, it's necessary to inverse-transform the predicted prices back to their original scale. This is done using the inverse_transform method of the scaler.

Mean Squared Error: 0.0006048302769427443. The MSE of 0.0006048302769427443 is relatively low, indicating that the LSTM model predicts Dogecoin prices with a small average squared difference. 






IV. References:
Time Series forecasting and Recurrent neural network.  Tensorflow Core (no date) TensorFlow. Available at: https://www.tensorflow.org/tutorials/structured_data/time_series#recurrent_neural_network (Accessed: 22 November 2023). 

OpenAI. (2023). GPT-3.5, a product of OpenAI. Accessed on November, 2023. Available at: https://chat.openai.com.
