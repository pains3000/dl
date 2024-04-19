import tensorflow as tf

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Display shapes of the datasets
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Visualize a sample image
import matplotlib.pyplot as plt
plt.imshow(X_train[2], cmap=plt.cm.binary)
plt.show()

# Normalize the pixel values
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

# Visualize a sample normalized image
print("Sample normalized image:")
plt.imshow(X_train[2], cmap=plt.cm.binary)
plt.show()

# Define the CNN model
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM

inputs = KL.Input(shape=(28, 28, 1))
c = KL.Conv2D(32, (3, 3), padding="valid", activation=tf.nn.relu)(inputs)
m = KL.MaxPool2D((2, 2), (2, 2))(c)
d = KL.Dropout(0.5)(m)
c = KL.Conv2D(64, (3, 3), padding="valid", activation=tf.nn.relu)(d)
m = KL.MaxPool2D((2, 2), (2, 2))(c)
d = KL.Dropout(0.5)(m)
c = KL.Conv2D(128, (3, 3), padding="valid", activation=tf.nn.relu)(d)
f = KL.Flatten()(c)
outputs = KL.Dense(10, activation=tf.nn.softmax)(f)

model = KM.Model(inputs, outputs)
model.summary()

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train.reshape(-1, 28, 28, 1), y_train, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test.reshape(-1, 28, 28, 1), y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)
