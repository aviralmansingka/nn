import numpy as np
from numpy import exp


def generate_data(w1, w2):

    X1 = w1 * np.random.randn(100, 4) + w1
    X2 = w2 * np.random.randn(100, 4) + w2

    X = np.append(X1, X2, axis=0)

    Y1 = np.ones((100, 1))
    Y2 = np.zeros((100, 1))

    Y = np.append(Y1, Y2, axis=0)

    return np.append(X, Y, axis=1)


def init_weights(layers=[]):

    assert layers != []

    W = []

    for i in range(len(layers)-1):

        w = np.random.randn(layers[i+1], layers[i])
        W.append(w)

    return W

def model(X, W):

    Y = []
    Z = []

    input = X.T

    for w in W:

        z = np.dot(w, input)
        print("z:", z.shape)
        Z.append(z)

        y = 1 / (1+exp(-z))
        print("y:", y.shape)
        Y.append(y)

        input = y

    return np.array(Y), np.array(Z)


def cal_grad(W, Y, Z, Y_true):

    # last layer
    Y2_grad = -1 * np.sum(Y_true - Y[-1])
    Z_grad = np.sum(Y[-1] * (1 - Y[-1])) * Y2_grad

    import ipdb; ipdb.set_trace()
    # second last layer
    Y1_grad = W[-1] * Z_grad



if __name__ == '__main__':

    data = generate_data(4, -4)
    assert data.shape == (200,5)

    X = data[:, :4]
    Y_true = data[:, 4:]

    W = init_weights([4, 3, 1])

    Y, Z = model(X, W)
    print("Y:", Y[0].shape, Y[1].shape)
    print("Z:", Z[0].shape, Z[1].shape)

    cal_grad(W, Y, Z, Y_true)
