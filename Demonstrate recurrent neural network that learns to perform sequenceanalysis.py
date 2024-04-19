import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from matplotlib import pyplot

# Disable TensorFlow Datasets progress bar
tfds.disable_progress_bar()

# Function to plot graphs
def plot_graphs(history, metric):
    pyplot.plot(history.history[metric])
    pyplot.plot(history.history['val_'+metric], '')
    pyplot.xlabel("Epochs")
    pyplot.ylabel(metric)
    pyplot.legend([metric, 'val_'+metric])

# Load IMDb movie reviews dataset
dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

# Display dataset element specifications
train_dataset.element_spec

# Display example text and labels
for example, label in train_dataset.take(5):
    print('text: ', example.numpy())
    print('label: ', label.numpy())

# Configure dataset for training
BUFFER_SIZE = 10000
BATCH_SIZE = 64
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Display example texts and labels after batching
for example, label in train_dataset.take(1):
    print('texts: ', example.numpy()[:3])
    print('labels: ', label.numpy()[:3])

# Define vocabulary size and create text encoder
VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))
vocab = np.array(encoder.get_vocabulary())
vocab[:20]

# Display encoded example texts
encoded_example = encoder(example)[:3].numpy()
encoded_example

for n in range(3):
    print("Original: ", example[n].numpy())
    print("Round-trip: ", " ".join(vocab[encoded_example[n]]))
    print()

# Define the model architecture
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=64, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

print([layer.supports_masking for layer in model.layers])

# Compile the model
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=test_dataset,
    validation_steps=30
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

# Plot accuracy and loss graphs
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.ylim(0, None)
