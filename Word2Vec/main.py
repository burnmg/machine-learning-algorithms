from utils import collect_data
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
import numpy as np
from negative_sampling_model import NegativeSamplingWord2VecEmbedding

window_size = 3
vector_dim = 300
epochs = 5

valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
vocab_size = 10000

embedding_dim = 300


data, count, dictionary, reverse_dictionary = collect_data(vocabulary_size=vocab_size, max_words_len=100)


sampling_table = sequence.make_sampling_table(vocab_size)
print("sampling table:", sampling_table.shape)
couples, labels = sequence.skipgrams(data, vocab_size, window_size=window_size, sampling_table=sampling_table)
# word_target, word_context = zip(*couples)
# word_target = np.array(word_target, dtype="int32")
# word_context = np.array(word_context, dtype="int32")

print(couples[:10], labels[:10])
train_ds = tf.data.Dataset.from_tensor_slices(
    (couples, labels)).shuffle(10000).batch(32)



# Create the model
model = NegativeSamplingWord2VecEmbedding(vocab_size, embedding_dim)

# Training
loss_object = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.RMSprop()

train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.s(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(inputs, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    # train_accuracy(labels, predictions)


@tf.function
def test_step(inputs, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(inputs, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    # test_accuracy(labels, predictions)

EPOCHS = 5

for epoch in range(EPOCHS):
      # Reset the metrics at the start of the next epoch
      train_loss.reset_states()
      # train_accuracy.reset_states()
      test_loss.reset_states()
      # test_accuracy.reset_states()

      for inputs, labels in train_ds:
        train_step(inputs, labels)

      # for test_images, test_labels in test_ds:
      #   test_step(test_images, test_labels)

      # template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
      template = 'Epoch {}, Loss: {}'
      print(template.format(epoch+1, train_loss.result()))


