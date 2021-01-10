from keras import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, Flatten, AveragePooling2D, MaxPooling2D
from keras.layers import regularizers
from keras.optimizers import adam
import matplotlib.pyplot as plt
from keras import utils
import numpy as np
import pickle
init_print = "=====>>"

(X_train, y_train), (X_test, y_test) = mnist.load_data()

### See the shape and the data in dataset
print(init_print, "X_train.shape:", X_train.shape)
#plt.imshow(X_train[100])
#print(y_train[100])
#plt.show()

### Transform the input X from (n, Width, Height) to (n, Width, Height, Depth)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
### change the type and normalise
print(X_train.dtype)
X_train = np.float32(X_train)
X_test = np.float32(X_test)
print(X_train.dtype)
X_train = X_train / 255
X_test = X_test / 255

### Transform labels Y into categorical data
y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)
print(y_train.shape)

### Configure the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(X_train.shape[1],X_train.shape[2],1)))
model.add(MaxPooling2D(pool_size=(2,2)))
print(model.output_shape)
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(X_train, y_train, batch_size=32, epochs=10)
print("The model has successfully trained")


model.save('mnist_CNN.h5')
print("Saving the model as mnist.h5")

score = model.evaluate(X_test, y_test, verbose=1)
print(score)

"""
New Things to learn in CNN

batch_size:

number of filters as the first parameter of Conv2D = in CNNs, filters are designed by the network to convolve on the image.
This filters could work like edge detection kernels, bluring kernels, etc. The fact is that these kernels are leant through training process
and the only thing we should do is to define the number of them. Remember that the input data of the CNN is 3D and has Width, Height, and Depth.
If the input is an RGB image, it might have the dimensions of (28,28,3). Also, if we set the size of each those filters as (3*3) or (5*5) and the number of them as M, then we will have
M filters that has width of 3, height of 3, and depth of 3 (since the image has depth of 3). M filter of (3,3,3)
In the next layer, if we set the number of filter as 64, we will have 64 kernels that each of them are (26*26*32).

kernel = filter

need to define third dimension for our data

In CNN architecture, the layers close to input has less number of filters (ex. 32), and the layers close to output has
bigger number of filters (ex. 128, 256).

softmax activation function works as probability
"""

