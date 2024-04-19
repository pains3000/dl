from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

# Generate synthetic data for training
X, Y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)

# Scale the input features
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)

# Define the neural network model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

# Train the model
model.fit(X, Y, epochs=500)

# Generate new synthetic data for prediction
X_new, Y_real = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1)
X_new = scalar.transform(X_new)

# Predict new data
Y_new = (model.predict(X_new) > 0.5).astype("int32")

# Print predictions
for i in range(len(X_new)):
    print("X=%s, Predicted=%s, Desired=%s" % (X_new[i], Y_new[i], Y_real[i]))
