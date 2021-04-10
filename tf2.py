import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD, Adam,RMSprop
from tensorflow.keras.metrics import SparseCategoricalAccuracy, Mean

n_sample = 1000
train_x = np.random.normal(0, 1, size=(n_sample, 1)).astype(np.float32)
train_x_noise = train_x + 0.2*np.random.normal(0, 1, size=(n_sample, 1)).astype(np.float32)
train_y = (train_x_noise > 0).astype(np.float32)

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(train_x, train_y)
ax.grid()
plt.show()

train_ds = tf.data.Dataset.from_tensor_slices((train_x_noise, train_y))
train_ds = train_ds.shuffle(n_sample).batch(8)

model = Sequential()
model.add(Dense(units=2, activation='softmax'))

loss_object = SparseCategoricalCrossentropy()
optimizer = SGD(learning_rate=1)

train_loss = Mean()
train_acc = SparseCategoricalAccuracy()

EPOCHS = 10

for epoch in range(EPOCHS):
    for x, y in train_ds:
        with tf.GradientTape() as tape:
            predictinos = model(x)
            loss = loss_object(y, predictinos)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_acc(y, predictinos)
    print('Epoch: ', epoch + 1)
    template = 'Train Loss: {:.4f}\t Train Accuracy: {:.2f}%\n'
    print(template.format(train_loss.result(), train_acc.result()*100))

    train_loss.reset_states()
    train_acc.reset_states()

model1 = Sequential()
model1.add(Dense(units=2, activation='softmax'))
model1.compile(loss=loss_object, optimizer=optimizer, metrics=['accuracy'])
model1.fit(train_x_noise, train_y, epochs=10)