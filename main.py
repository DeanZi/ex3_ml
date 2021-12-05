import sys
import numpy as np


sigmoid = lambda x: 1 / (1 + np.exp(-x))
dsigmoid = lambda x: sigmoid(x)*(1-sigmoid(x))

def softmax(output):
    output = output - np.max(output, axis = 1).reshape(output.shape[0], 1)
    return np.exp(output) / np.sum(np.exp(output), axis = 1).reshape(output.shape[0], 1)

class NeuralNetwork:
    def __init__(self, epochs, learning_rate, num_of_layers):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.input_size = 784
        self.output_size = 10
        self.num_of_layers = num_of_layers
        self.weights, self.biases = self.init_parameters()

    def init_parameters(self):
        weights = []
        biases = []
        for layer in range(self.num_of_layers):
            if layer == 0:
                weights.append(np.random.randn(self.input_size, 256))
            elif layer == self.num_of_layers - 1:
                weights.append(np.random.randn(weights[layer-1].shape[1], self.output_size))
            else:
                weights.append(np.random.randn(weights[layer-1].shape[1], int(256/(2 * layer))))
            biases.append(np.random.randn(weights[layer].shape[1]))

        return weights, biases

    def feedforward(self, train_x):
        z_values = []
        h_values = []
        for layer in range(self.num_of_layers):
            if layer == 0:
                z_values.append(train_x.dot(self.weights[layer]) + self.biases[layer])
                h_values.append(sigmoid(z_values[layer]))
            if layer == self.num_of_layers - 1:
                z_values.append(h_values[layer-1].dot(self.weights[layer]))
                h_values.append(softmax(z_values[layer]))
            else:
                z_values.append(h_values[layer-1].dot(self.weights[layer]))
                h_values.append(sigmoid(z_values[layer]))
        return h_values[self.num_of_layers - 1]


    def backpropagation(self, train_y, y_hats, layer_id):
        dl_dz = {}
        dl_dw = {}
        #for layer in range(self.num_of_layers - 1, -1, -1):
        if layer_id == self.num_of_layers - 1:
            dl_dz[layer] = y_hats - train_y
            dl_dw[layer] = np.dot(dl_dz[layer], y_hats)
            return self.backpropagation(train_y, y_hats, layer_id - 1)
        else:
            dl_dz[layer] =

        dl_db = dl_dz





def receive_data(train_x, train_y, test_x):
    train_x = np.loadtxt(train_x, max_rows=500)
    train_y = np.loadtxt(train_y, max_rows=500)
    test_x = np.loadtxt(test_x, max_rows=500)
    return train_x, train_y, test_x


if __name__ == '__main__':
    train_x, train_y, test_x = receive_data(sys.argv[1], sys.argv[2], sys.argv[3])
    network = NeuralNetwork(epochs=50, num_of_layers=3, learning_rate=5e-3)
    network.backpropagation()