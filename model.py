from collections import OrderedDict
from pooling import Pooling
from convolution import Convolution
from fullyConnected import FullyConnected
from utils import *

class Model:
    def __init__(self):
        np.random.seed(42)
        self.params = {
            'W1': np.random.randn(16, 3, 3, 3) * 0.01,  # Convolution weights (FN, C, FH, FW)
            'b1': np.zeros(16),  # Convolution bias
            'W2': np.random.randn(64 * 64 * 16, 64) * 0.01,  # Fully connected weights (flattened output of pooling to hidden layer)
            'b2': np.zeros(64),
            'W3': np.random.randn(64, 1) * 0.01,  # Output weights (hidden layer to output)
            'b3': np.zeros(1)
        }
        self.layers = OrderedDict()
        self.layers['conv1'] = Convolution(self.params['W1'], self.params['b1'])
        self.layers['pool1'] = Pooling(2, 2, stride=2)
        self.layers['fc1'] = FullyConnected(self.params['W2'], self.params['b2'])
        self.layers['fc2'] = FullyConnected(self.params['W3'], self.params['b3'])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return sigmoid(x)

    def loss(self, x, t):
        y = self.predict(x)
        return binary_cross_entropy_error(y, t)

    def gradient(self, x, t):
        # Forward
        y = self.predict(x)

        # Backward (gradient calculation)
        dout = (y - t) / t.shape[0]
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # Set gradients to params
        grads = {
            'W1': self.layers['conv1'].dW, 'b1': self.layers['conv1'].db,
            'W2': self.layers['fc1'].dW, 'b2': self.layers['fc1'].db,
            'W3': self.layers['fc2'].dW, 'b3': self.layers['fc2'].db
        }
        return grads