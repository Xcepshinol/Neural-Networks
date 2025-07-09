import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('MNIST_CSV/mnist_train.csv')
test = pd.read_csv('MNIST_CSV/mnist_test.csv')

data = np.array(data)
test = np.array(test)
m, n = data.shape
m1, n1 = test.shape

data_train = data.T
data_y = data_train[0]
data_x = data_train[1:n]
data_x = data_x/255

data_test = test.T
data_yt = data_train[0]
data_xt = data_train[1:n1]
data_xt = data_xt/255

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max()+1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X)+b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1)+b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def backwards_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size

    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2)

    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2

def get_accuracy(prediction, Y):
    return np.sum(prediction == Y)/Y.size

def get_predictions(A2):
    return np.argmax(A2, 0)

def gradient_desc(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backwards_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            print("Accuracy :", get_accuracy(get_predictions(A2), Y))

    return W1, b1, W2, b2

try:
    data = np.load("MNIST_CSV/stored_data.npz")
    W1, b1, W2, b2 = data['W1'], data['b1'], data['W2'], data['b2']

except:
    W1, b1, W2, b2 = gradient_desc(data_x, data_y, 500, 0.1)
    np.savez("stored_data.npz", W1=W1, b1=b1, W2=W2, b2=b2)

index = 30

_, _, _, A2 = forward_prop(W1, b1, W2, b2, data_xt[:, index, None])
prediction = get_predictions(A2)
label = data_yt[index]

print("Prediction: ", prediction)
print("Label: ", label)

current_image = data_xt[:, index, None].reshape((28, 28)) * 255
plt.gray()
plt.imshow(current_image, interpolation='nearest')
plt.show()