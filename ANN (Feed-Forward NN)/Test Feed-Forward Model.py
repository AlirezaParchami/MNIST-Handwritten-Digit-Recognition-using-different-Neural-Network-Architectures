from keras.datasets import mnist
from keras.models import load_model
import keras.utils as utils
import numpy as np

# read data set
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_test = np.reshape(X_test, (X_test.shape[0],1,28*28))
input_shape = (1, 28*28)
num_classes = 10

# Convert the X and Y into desirable form
Y_test = utils.to_categorical(Y_test, num_classes=num_classes)
X_test = X_test.astype('float32')
X_test = X_test / 255

model = load_model('mnist.h5')
print("The model loaded successfully")

prediction = model.predict(X_test, verbose=1)

#prediction[prediction > 0.5] = 1
#prediction[prediction <= 0.5] = 0
prediction = np.where(prediction>0.5, 1 , 0)

def column(mat, index):
    return [row[index] for row in mat]

def confusion_matrix_index(actual, predicted):
    true_pos = len([1 for j in range(0, len(actual)) if actual[j] == predicted[j] and predicted[j] == 1])
    true_neg = len([1 for j in range(0, len(actual)) if actual[j] == predicted[j] and predicted[j] == 1])
    false_pos = len([1 for j in range(0, len(actual)) if actual[j] != predicted[j] and predicted[j] == 1])
    false_neg = len([1 for j in range(0, len(actual)) if actual[j] != predicted[j] and predicted[j] == 0])
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_neg + false_pos)
    precision = (true_pos) / (true_pos + false_pos)
    recall = (true_pos) / (true_pos + false_neg)
    F1_score = 2*(recall*precision) / (recall + precision)
    return [accuracy, precision, recall, F1_score]

confusion_mat = [["Accuracy", "Precision", "Recall", "F1-Score"]]
for i in range(0, 10):
    confusion_mat.append(["number "+str(i), confusion_matrix_index(column(Y_test,i), column(prediction, i))])

for i in range(0, len(confusion_mat)):
    print(confusion_mat[i])
#from sklearn.metrics import confusion_matrix
#
#def my_confusion_matrix(actual, predicted):
#    true_positives = len([a for a, p in zip(actual, predicted) if a.all() == p.all() and p == 1])
#    true_negatives = len([a for a, p in zip(actual, predicted) if a == p and p == 0])
#    false_positives = len([a for a, p in zip(actual, predicted) if a != p and p == 1])
#    false_negatives = len([a for a, p in zip(actual, predicted) if a != p and p == 0])
#    return "[[{} {}]\n  [{} {}]]".format(true_negatives, false_positives, false_negatives, true_positives)
#
#print("my Confusion Matrix A:\n", my_confusion_matrix(Y_test, prediction))
#print("sklearn Confusion Matrix A:\n", confusion_matrix(Y_test, prediction))

score = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
