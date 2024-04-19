from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler

# Generate synthetic regression data for training
X, Y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=1)

# Scale the input features and target variable
scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
scalarX.fit(X)
scalarY.fit(Y.reshape(100,1))
X = scalarX.transform(X)
Y = scalarY.transform(Y.reshape(100,1))

# Define the neural network model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))  # Linear activation for regression
model.compile(loss='mse', optimizer='adam')

# Train the model
model.fit(X, Y, epochs=1000, verbose=0)

# Generate new synthetic data for prediction
X_new, _ = make_regression(n_samples=3, n_features=2, noise=0.1, random_state=1)
X_new = scalarX.transform(X_new)

# Predict new data
Y_new = model.predict(X_new)

# Print predictions
for i in range(len(X_new)):
    print("X=%s, Predicted=%s" % (X_new[i], Y_new[i]))
