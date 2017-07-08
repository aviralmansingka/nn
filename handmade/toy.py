import numpy as np
from numpy import exp
import matplotlib.pyplot as plt


def generate_data(w1, w2, shape):
    x1 = np.random.randn(shape[0]/2, shape[1]-1) + w1
    x2 = np.random.randn(shape[0]/2, shape[1]-1) + w2

    x = np.append(x1, x2, axis=0)

    y1 = np.ones((shape[0]/2, 1))
    y2 = np.zeros((shape[0]/2, 1))

    y = np.append(y1, y2, axis=0)

    return np.append(x, y, axis=1)


def init_weights(layers=[]):
    W = []
    for i in range(len(layers)-1):
        w = 0.01 * np.random.randn(layers[i], layers[i+1])
        W.append(w)

    return W

def model(x, W):
    Z = []
    Y = []
    input = x
    for w in W:
        z = np.dot(input, w)
        y = 1/(1+exp(-z))

        Z.append(z)
        Y.append(y)
        input = y


    return Z, Y


def cal_grads(W, x, Y, Z, y_true):

    # last layer
    loss = -1 * np.sum((y_true - Y[-1])**2)

    y1_grad = -1 * np.sum((y_true - Y[-1]), axis=0) / 10
    z1_grad = np.sum(Y[-1] * (1-Y[-1]) * -1 * (y_true - Y[-1]), axis=0)
    w1_grad = (np.sum(Y[-2], axis=0) * z1_grad).reshape(3,1)

    y0_grad = z1_grad * W[-1]
    z0_grad = np.sum(Y[-2] * (1-Y[-2]) * -1 * y1_grad, axis=0).reshape(3,1)
    w0_grad = (np.sum(x,axis=0) * z0_grad).T

    return [w0_grad, w1_grad], loss


def update_weights(W, grads, lr=0.05):

    for w, grad in zip(W, grads):
        assert w.shape == grad.shape
        w -= lr * grad

    return W


if __name__ == '__main__':

    input_dim = 3
    number_rows = 100
    data_shape = (number_rows, input_dim)

    data = generate_data(1, -1, shape=data_shape)
    assert data.shape == data_shape

    negative = data[data[:, 2] == 0]
    positive = data[data[:, 2] == 1]

    np.random.shuffle(data)
    x = data[:, :input_dim-1]
    y_true = data[:, input_dim-1:]

    W = init_weights([input_dim-1, 3, 1])

    for i in range(1000):

        Z, Y = model(x, W)
        grads, loss = cal_grads(W, x, Y, Z, y_true)
        W = update_weights(W, grads, lr=0.1)

        pred = np.round(Y[-1])

        a = pred.flatten() == y_true.flatten()
        result = float(np.sum(a)) / number_rows

        if i % 100 == 0:
            print(i, result, loss)

    print(W)
