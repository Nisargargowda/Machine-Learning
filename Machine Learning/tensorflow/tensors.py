#First neural network
#Train, evaluate and predict with model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#Load the dataset
mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)

# Normalise the dataset: from 0, 255 -> 0, 1
x_test, x_train = x_test / 255.0, x_train / 255.0

# Model
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10),
])

print(model.summary())

# loss and optimiser
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)

#Training
batch_size = 64
epochs = 5

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)

#evaluate
model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)

#Predictions

#1. option: build a new model with Softmax function
probability_model1 = keras.models.Sequential([
    model,
    keras.layers.Softmax()
])

predictions = probability_model1(x_test)
pred0 = predictions[0]
print(pred0)

# Use np.argmax to get label with highest probability
label0 = np.argmax(pred0)
print(label0)