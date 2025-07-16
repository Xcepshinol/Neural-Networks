import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


try:
    data = np.load("QuickDraw/combined_labeled_dataset.npy", encoding='latin1', allow_pickle=True)

except:
    files = [
    ("QuickDraw/full_numpy_bitmap_airplane.npy", 0),
    ("QuickDraw/full_numpy_bitmap_ant.npy", 1),
    ("QuickDraw/full_numpy_bitmap_apple.npy", 2),
    ("QuickDraw/full_numpy_bitmap_banana.npy", 3),
    ("QuickDraw/full_numpy_bitmap_fish.npy", 4),
    ("QuickDraw/full_numpy_bitmap_snake.npy", 5),
    ("QuickDraw/full_numpy_bitmap_strawberry.npy", 6)
    ]

    combined_data = []

    for i, j in files:
        data = np.load(i, encoding='latin1', allow_pickle=True)[:40000]

        labels = np.full((data.shape[0], 1), j)
        labeled_data = np.hstack((labels, data))
        combined_data.append(labeled_data)

    final_dataset = np.vstack(combined_data)
    
    np.save("QuickDraw/combined_labeled_dataset.npy", final_dataset)
    data = np.load("QuickDraw/combined_labeled_dataset.npy", encoding='latin1', allow_pickle=True)

m, n = data.shape
np.random.shuffle(data)

dev_data = data[0:1000].T
dev_y = dev_data[0]
dev_x = dev_data[1:n]
dev_x = dev_x/255

test_data = data[1000:m].T
test_y = test_data[0]
test_x = test_data[1:n]
test_x = test_x/255

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))

def deriv_ReLU(Z):
    return Z > 0

def init_params():
    W1 = np.random.rand(32, 784) - 0.5
    b1 = np.random.rand(32, 1) - 0.5
    W2 = np.random.rand(32, 32) - 0.5
    b2 = np.random.rand(32, 1) - 0.5
    W3 = np.random.rand(7, 32) - 0.5
    b3 = np.random.rand(7, 1) - 0.5
    return W1, b1, W2, b2, W3, b3

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max()+1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def backwards_prop(Z1, A1, Z2, A2, Z3, A3, W3, W2, X, Y):
    m = Y.size

    one_hot_Y = one_hot(Y)
    dZ3 = A3 - one_hot_Y
    dW3 = 1/m * dZ3.dot(A2.T)
    db3 = 1/m * np.sum(dZ3)

    dZ2 = W3.T.dot(dZ3) * deriv_ReLU(Z2) 
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2)

    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1) 
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1)

    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha*dW1
    b1 = b1 - alpha*db1
    W2 = W2 - alpha*dW2
    b2 = b2 - alpha*db2
    W3 = W3 - alpha*dW3
    b3 = b3 - alpha*db3
    return W1, b1, W2, b2, W3, b3

def get_accuracy(prediction, Y):
    return np.sum(prediction == Y)/Y.size

def get_predictions(A3):
    return np.argmax(A3, 0)

def gradient_desc(X, Y, iterations, alpha):
    W1, b1, W2, b2, W3, b3 = init_params()

    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backwards_prop(Z1, A1, Z2, A2, Z3, A3, W3, W2, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)

        if i % 50 == 0:
            print("Accuracy: ", get_accuracy(get_predictions(A3), Y))

    return W1, b1, W2, b2, W3, b3


try:
    data = np.load("QuickDraw/QDstored_data.npz")
    W1, b1, W2, b2, W3, b3 = data['W1'], data['b1'], data['W2'], data['b2'], data['W3'], data['b3']

except:
    W1, b1, W2, b2, W3, b3 = gradient_desc(test_x, test_y, 500, 0.5)
    np.savez("QuickDraw/QDstored_data.npz", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)

index = 30

_, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, test_x[:, index, None])
prediction = get_predictions(A3)
label = test_y[index]

print("Prediction: ", prediction)
print("Label: ", label)

current_image = test_x[:, index, None].reshape((28, 28)) * 255
plt.gray()
plt.imshow(current_image, interpolation='nearest')
plt.show()