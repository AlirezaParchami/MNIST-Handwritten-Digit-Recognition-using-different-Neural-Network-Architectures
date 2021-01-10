import numpy as np
from keras import utils, optimizers
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.layers import Dropout, regularizers
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print(X_train.shape, Y_train.shape)

X_train = np.reshape(X_train, (X_train.shape[0],1,28*28))
X_test = np.reshape(X_test, (X_test.shape[0],1,28*28))
input_shape = (1, 28*28)
num_classes = 10
# convert class vectors in to binary class matrices
Y_train = utils.to_categorical(Y_train, num_classes=num_classes)
Y_test = utils.to_categorical(Y_test, num_classes=num_classes)

# Normalise images
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255
X_test = X_test / 255


# Create Model :)
model = Sequential();

model.add(Flatten(input_shape=input_shape))
model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.005)))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adadelta(), metrics=['acc'])
print(model.summary())
hist = model.fit(X_train, Y_train, batch_size=128, epochs=10)
print("The model has successfully trained")

losses = pd.DataFrame(hist.history['acc'])
losses.plot()
plt.draw()


model.save('mnist.h5')
print("Saving the model as mnist.h5")

prediction = model.predict(X_test, verbose=1)
prediction = utils.to_categorical(prediction, num_classes=num_classes)
print(classification_report(Y_test, prediction))