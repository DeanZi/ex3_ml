import sys
import numpy as np


sigmoid = lambda x: 1 / (1 + np.exp(-x))
dsigmoid = lambda x: sigmoid(x)*(1-sigmoid(x))

def softmax(output):
    output = output - np.max(output, axis = 1).reshape(output.shape[0], 1)
    return np.exp(output) / np.sum(np.exp(output), axis = 1).reshape(output.shape[0], 1)

class NeuralNetwork:
    def __init__(self, epochs, learning_rate, num_of_layers, train_x, train_y):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.input_size = 784
        self.output_size = 10
        self.num_of_layers = num_of_layers
        self.weights, self.biases = self.init_parameters()
        self.z_values = []
        self.h_values = []
        self.y_hat = 0
        self.dl_dz = {}
        self.dl_dw = {}
        self.dl_db = {}
        self.train_x = train_x
        self.train_y = train_y

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

    def feedforward(self, input_example):
        for layer in range(self.num_of_layers):
            if layer == 0:
                self.z_values.append(input_example.dot(self.weights[layer]) + self.biases[layer])
                self.h_values.append(sigmoid(self.z_values[layer]))
            if layer == self.num_of_layers - 1:
                self.z_values.append(self.h_values[layer-1].dot(self.weights[layer]))
                self.h_values.append(softmax(self.z_values[layer]))
            else:
                self.z_values.append(self.h_values[layer-1].dot(self.weights[layer]))
                self.h_values.append(sigmoid(self.z_values[layer]))
        self.y_hat = self.h_values[self.num_of_layers - 1].index(min(self.h_values[self.num_of_layers - 1]))
        return self.y_hat


    def backpropagation(self, target):
        for layer_id in range(self.num_of_layers-1, -1, -1):
            if layer_id == self.num_of_layers - 1:
                self.dl_dz[layer_id] = self.y_hat - target
                self.dl_dw[layer_id] = np.dot(self.dl_dz[layer_id], self.y_hat)
            elif layer_id > 0:
                self.dl_dz[layer_id] = np.dot(self.dl_dz[layer_id + 1], self.weights[layer_id+1]) * dsigmoid(self.z_values[layer_id])
                self.dl_dw[layer_id] = np.dot(self.dl_dz[layer_id], self.h_values[layer_id-1])
            else:
                self.dl_dz[layer_id] = np.dot(self.dl_dz[layer_id + 1], self.weights[layer_id+1]) * dsigmoid(self.z_values[layer_id])
                self.dl_dw[layer_id] = np.dot(self.dl_dz[layer_id], self.train_x)
        self.dl_db = self.dl_dz

        for index, weight in enumerate(self.weights):
            weight = weight - self.learning_rate * self.dl_dw[index]
            self.weights[index] = weight

    def shuffle(self):
        index = [x for x in range(self.train_x.shape[0])]
        np.random.shuffle(index)
        self.train_x = self.train_x[index]
        self.train_y = self.train_y[index]



    def train(self):
        for _ in range(self.epochs):
            self.shuffle()
            for input in self.train_x:
                self.feedforward(input)
                self.backpropagation()

    def predict(self, test_x):
        for example in test_x:
            print(self.feedforward(example))






def receive_data(train_x, train_y, test_x):
    train_x = np.loadtxt(train_x, max_rows=500)
    train_y = np.loadtxt(train_y, max_rows=500)
    test_x = np.loadtxt(test_x, max_rows=500)
    return train_x, train_y, test_x


if __name__ == '__main__':
    train_x, train_y, test_x = receive_data(sys.argv[1], sys.argv[2], sys.argv[3])
    network = NeuralNetwork(epochs=50, num_of_layers=3, learning_rate=5e-3, train_x=train_x, train_y=train_y)
    network.train()
    network.predict(test_x)