import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Measure the start time
start_time = time.time()

# Load and preprocess data
data = pd.read_csv('/content/data/SP500_1d_data.csv')
data['Target'] = data['Close'].shift(-1)
data.dropna(inplace=True)

dates = data['Date']
X = data[['Open', 'High', 'Low', 'Close']].values
y = data['Target'].values

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X_scaled, y_scaled, dates, test_size=0.2, random_state=42)

# Define the neural network
class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        self.W1 = np.random.randn(input_size, hidden_size1) * 0.01
        self.b1 = np.zeros((1, hidden_size1))
        self.W2 = np.random.randn(hidden_size1, hidden_size2) * 0.01
        self.b2 = np.zeros((1, hidden_size2))
        self.W3 = np.random.randn(hidden_size2, output_size) * 0.01
        self.b3 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        output = self.z3
        return output

    def backward(self, X, y, output):
        d_loss_output = 2 * (output - y) / y.size
        d_loss_a2 = d_loss_output.dot(self.W3.T) * self.relu_derivative(self.z2)
        d_loss_a1 = d_loss_a2.dot(self.W2.T) * self.relu_derivative(self.z1)

        dW3 = self.a2.T.dot(d_loss_output)
        db3 = np.sum(d_loss_output, axis=0, keepdims=True)
        dW2 = self.a1.T.dot(d_loss_a2)
        db2 = np.sum(d_loss_a2, axis=0, keepdims=True)
        dW1 = X.T.dot(d_loss_a1)
        db1 = np.sum(d_loss_a1, axis=0, keepdims=True)

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        self.losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            loss = np.mean((output - y) ** 2)
            self.losses.append(loss)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        return self.forward(X)

# Train the neural network
input_size = X_train.shape[1]
hidden_size1 = 64
hidden_size2 = 32
output_size = 1
learning_rate = 0.001
epochs = 2000

nn = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)
nn.train(X_train, y_train, epochs, learning_rate)

# Make predictions and print results
predictions = nn.predict(X_test)
predictions = scaler_y.inverse_transform(predictions)

# Get actual 'Close' prices for the test set
actual_prices = data.loc[data.index.isin(dates_test.index), 'Close'].values

# Measure the end time
end_time = time.time()
runtime = end_time - start_time

print('Date\t\t\tActual Price\tPredicted Price')
for date, actual_price, predicted_price in zip(dates_test, actual_prices, predictions):
    print(f'{date}\t{actual_price:.2f}\t{predicted_price[0]:.2f}')

# Plot the results and save as an image
plt.figure(figsize=(14, 7))
plt.plot(dates_test, actual_prices, label='Actual Prices', color='red')
plt.plot(dates_test, predictions, label='Predicted Prices', color='blue')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('SP500 Price Predictions')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('sp500_price_predictions.png')

print(f'Runtime: {runtime:.2f} seconds')
