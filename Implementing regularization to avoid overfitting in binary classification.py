from matplotlib import pyplot
from sklearn.datasets import make_moons
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2, l1_l2

# Generate moon-shaped dataset
X, Y = make_moons(n_samples=100, noise=0.2, random_state=1)

# Split dataset into train and test sets
n_train = 30
trainX, testX = X[:n_train, :], X[n_train:],
trainY, testY = Y[:n_train], Y[n_train:]

print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)

# Model without regularization
model = Sequential()
model.add(Dense(500, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train model
history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100)

# Plot accuracy
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()

# Model with L2 regularization
model = Sequential()
model.add(Dense(500, input_dim=2, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train model
history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100)

# Plot accuracy
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()

# Model with L1_L2 regularization
model = Sequential()
model.add(Dense(500, input_dim=2, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train model
history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100)

# Plot accuracy
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()
