import sys
import numpy as np
from scipy.special import expit


sigmoid = lambda x: expit(x)#1 / (1 + np.exp(-x))
dsigmoid = lambda x: sigmoid(x)*(1-sigmoid(x))
#TODO : check correctness of softmax
def softmax(output):
    output = output - np.max(output)
    return np.exp(output) / np.sum(np.exp(output))

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
            elif layer == self.num_of_layers - 1:
                self.z_values.append(np.dot(self.h_values[layer-1], self.weights[layer]) + self.biases[layer])
                self.h_values.append(softmax(self.z_values[layer]))
            else:
                self.z_values.append(np.dot(self.h_values[layer-1].T, self.weights[layer]) + self.biases[layer])
                self.h_values.append(sigmoid(self.z_values[layer]))
        y_hat = self.h_values[self.num_of_layers - 1].tolist().index(min(self.h_values[self.num_of_layers - 1]))

        return y_hat


    def backpropagation(self, target, input):
        for layer_id in range(self.num_of_layers-1, -1, -1):
            if layer_id == self.num_of_layers - 1:
                self.dl_dz[layer_id] = self.h_values[layer_id] - target
                self.dl_dw[layer_id] = np.dot(self.dl_dz[layer_id], self.h_values[layer_id])
            elif layer_id > 0:
                self.dl_dz[layer_id] = np.dot(self.weights[layer_id+1], self.dl_dz[layer_id + 1]) * dsigmoid(self.z_values[layer_id])
                self.dl_dz[layer_id] = self.dl_dz[layer_id].reshape(self.dl_dz[layer_id].shape[0], 1)
                self.h_values[layer_id - 1] = self.h_values[layer_id - 1].reshape(self.h_values[layer_id - 1].shape[0], 1)
                self.dl_dw[layer_id] = np.dot(self.h_values[layer_id-1], self.dl_dz[layer_id].T)
            else:
                self.dl_dz[layer_id] = np.dot(self.dl_dz[layer_id + 1].T, self.weights[layer_id+1].T) * dsigmoid(self.z_values[layer_id])
                input = input.reshape(input.shape[0], 1)
                self.dl_dw[layer_id] = np.dot(self.dl_dz[layer_id].T, input.T)
        self.dl_db = self.dl_dz

        for index, weight in enumerate(self.weights):
            if weight.shape == self.dl_dw[index].shape:
                weight = weight - self.learning_rate * self.dl_dw[index]
            else:
                weight = weight - self.learning_rate * self.dl_dw[index].T
            self.weights[index] = weight

        for index, bias in enumerate(self.biases):
            bias = bias - self.learning_rate * self.dl_db[index].T
            self.biases[index] = bias

    def shuffle(self):
        index = [x for x in range(self.train_x.shape[0])]
        np.random.shuffle(index)
        self.train_x = self.train_x[index]
        self.train_y = self.train_y[index]



    def train(self):
        for _ in range(self.epochs):
            self.shuffle()
            for input, target in zip(self.train_x, self.train_y):
                self.feedforward(input)
                self.backpropagation(target, input)

    def predict(self, test_x):
        results = []
        for example in test_x:
            results.append(str(self.feedforward(example)))
        return results






def receive_data(train_x, train_y, test_x):
    train_x = np.loadtxt(train_x)
    train_y = np.loadtxt(train_y)
    test_x = np.loadtxt(test_x)
    return train_x, train_y, test_x


if __name__ == '__main__':
    train_x, train_y, test_x = receive_data(sys.argv[1], sys.argv[2], sys.argv[3])
    network = NeuralNetwork(epochs=50, num_of_layers=3, learning_rate=5e-3, train_x=train_x, train_y=train_y)
    network.train()
    output_file = open('test_y', 'w')
    test_y = network.predict(test_x)
    for y in test_y:
        output_file.write(y + '\n')
    output_file.close()