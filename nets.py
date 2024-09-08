import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


class TwoLayerNet:
    '''
    A fully connected network with one hidden layer.
    '''

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        assert output_size == 1
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        self.a1 = np.dot(x, W1) + b1
        self.z1 = relu(self.a1)
        self.a2 = np.dot(self.z1, W2) + b2
        y = sigmoid(self.a2)
        return y

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.squeeze(y) > 0.5
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy


def load_network(saved_path):
    '''
    Load a neural network from the npz file.
    '''
    with np.load(saved_path, allow_pickle=True) as data_file:
        ws = data_file["arr_0"]
        bs = data_file["arr_1"]
    w1, w2 = ws[0], ws[1]
    b1, b2 = bs[0], bs[1]
    w1 = np.transpose(w1)
    w2 = np.transpose(w2)
    b1 = b1.flatten()
    b2 = b2.flatten()
    d0 = w1.shape[0]
    d1 = w1.shape[1]
    assert w1.shape[1] == w2.shape[0] and w2.shape[1] == 1
    assert len(b1) == d1 and len(b2) == 1
    network = TwoLayerNet(input_size=d0, hidden_size=d1, output_size=1)
    network.params['W1'] = w1
    network.params['W2'] = w2
    network.params['b1'] = b1
    network.params['b2'] = b2
    return network


def save_network(net: TwoLayerNet, saved_path):
    '''
    Save a neural network as a npz file.
    '''
    W1 = net.params['W1']
    W2 = net.params['W2']
    b1 = net.params['b1']
    b2 = net.params['b2']
    W1 = np.transpose(W1)
    W2 = np.transpose(W2)
    b1 = np.reshape(b1, (-1, 1))
    b2 = np.reshape(b2, (-1, 1))
    ws = [W1, W2]
    bs = [b1, b2]
    np.savez(saved_path, ws, bs)