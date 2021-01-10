from keras.datasets import mnist
from keras.models import load_model
import keras.utils as utils
import numpy as np

# read data set
(X_train, y_train), (X_test, y_test) = mnist.load_data()
model = load_model('mnist_CNN.h5')


X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_test = np.float32(X_test)
X_test /= 255
y_test = utils.to_categorical(y_test)

score = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])