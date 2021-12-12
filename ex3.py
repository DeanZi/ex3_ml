import math
import sys
from datetime import datetime

import numpy as np
import random
from scipy.special import expit

sigmoid = lambda x: expit(x)
dsigmoid = lambda x: sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return (x > 0) * x
def drelu(x):
    return x > 0

def softmax(output):
    output = output - np.max(output)
    return np.exp(output) / np.sum(np.exp(output))


class NeuralNetwork:
    def __init__(self, epochs, learning_rate, num_of_layers, train_x, train_y, normalize_weights=True):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.input_size = 784
        self.output_size = 10
        self.num_of_layers = num_of_layers
        self.normalize_weights = normalize_weights
        self.weights, self.biases = self.init_parameters(self.normalize_weights)
        self.z_values = {}
        self.h_values = {}
        self.dl_dz = {}
        self.dl_dw = {}
        self.dl_db = {}
        self.train_x = train_x
        self.train_y = train_y

    def init_parameters(self, normalize_weights):
        weights = []
        biases = []
        for layer in range(self.num_of_layers):
            if self.num_of_layers == 1:
                if normalize_weights:
                    weights.append(np.random.uniform(-1, 1, (self.input_size, self.output_size)))
                    biases.append(np.random.uniform(-1, 1, (weights[layer].shape[1], 1)))
                else:
                    weights.append(np.random.randn(self.input_size, self.output_size))
                    biases.append(np.random.randn(weights[layer].shape[1], 1))

            else:
                if layer == 0:
                    if normalize_weights:
                        weights.append(np.random.uniform(-1, 1, (self.input_size, 100)))
                    else:
                        weights.append(np.random.randn(self.input_size, 100))
                elif layer == self.num_of_layers - 1:
                    if normalize_weights:
                        weights.append(np.random.uniform(-1, 1, (weights[layer - 1].shape[1], self.output_size)))
                    else:
                        weights.append(np.random.randn(weights[layer - 1].shape[1], self.output_size))
                else:
                    if normalize_weights:
                        weights.append(np.random.uniform(-1, 1, (weights[layer - 1].shape[1], 100)))
                    else:
                        weights.append(np.random.randn(weights[layer - 1].shape[1], 100))
                if normalize_weights:
                    biases.append(np.random.uniform(-1, 1, (weights[layer].shape[1], 1)))
                else:
                    biases.append(np.random.randn(weights[layer].shape[1], 1))

        return weights, biases

    def feedforward(self, input_example):
        if self.num_of_layers == 1:
            self.z_values[0] = np.dot(self.weights[0].T, input_example) + self.biases[0]
            self.h_values[0] = softmax(self.z_values[0])

        else:
            for layer in range(self.num_of_layers):
                if layer == 0:
                    self.z_values[layer] = np.dot(self.weights[layer].T, input_example) + self.biases[layer]
                    self.h_values[layer] = relu(self.z_values[layer])
                elif layer == self.num_of_layers - 1:
                    self.z_values[layer] = np.dot(self.weights[layer].T, self.h_values[layer - 1]) + self.biases[layer]
                    self.h_values[layer] = softmax(self.z_values[layer])
                else:
                    self.z_values[layer] = np.dot(self.weights[layer].T, self.h_values[layer - 1]) + self.biases[layer]
                    self.h_values[layer] = relu(self.z_values[layer])
        max_h_val = 0
        y_hat = -1
        # print(self.h_values[self.num_of_layers - 1])
        for index, h_val in enumerate(self.h_values[self.num_of_layers - 1]):
            if h_val > max_h_val:
                max_h_val = h_val
                y_hat = index
        return y_hat

    def backpropagation(self, target, input):
        target = target.reshape(target.shape[0], 1)
        if self.num_of_layers == 1:
            self.dl_dz[0] = self.h_values[0] - target
            self.dl_dw[0] = np.dot(self.dl_dz[0], input.T)

        else:
            for layer_id in range(self.num_of_layers - 1, -1, -1):
                if layer_id == self.num_of_layers - 1:
                    self.dl_dz[layer_id] = self.h_values[layer_id] - target
                    self.dl_dw[layer_id] = np.dot(self.dl_dz[layer_id], self.h_values[layer_id - 1].T)

                elif layer_id > 0:
                    self.dl_dz[layer_id] = np.dot(self.weights[layer_id + 1], self.dl_dz[layer_id + 1]) * drelu(
                        self.z_values[layer_id])
                    self.dl_dw[layer_id] = np.dot(self.dl_dz[layer_id], self.h_values[layer_id - 1].T)
                else:
                    self.dl_dz[layer_id] = np.dot(self.weights[layer_id + 1], self.dl_dz[layer_id + 1]) * drelu(
                        self.z_values[layer_id])
                    self.dl_dw[layer_id] = np.dot(self.dl_dz[layer_id], input.T)
        self.dl_db = self.dl_dz

        for index, weight in enumerate(self.weights):
            if weight.shape == self.dl_dw[index].shape:
                weight = weight - self.learning_rate * self.dl_dw[index]
            else:
                weight = weight - self.learning_rate * self.dl_dw[index].T
            self.weights[index] = weight

        for index, bias in enumerate(self.biases):
            bias = bias - self.learning_rate * self.dl_db[index]
            self.biases[index] = bias

    def shuffle(self):
        shuffled_data = list(zip(self.train_x, self.train_y))
        random.Random(1).shuffle(shuffled_data)
        self.train_x, self.train_y = zip(*shuffled_data)

    def train(self):
        for _ in range(self.epochs):
            # print("Running epoch number ", _)
            # print(datetime.now())
            self.shuffle()
            loss = 0
            for input, target in zip(self.train_x, self.train_y):
                input = input.reshape(input.shape[0], 1)
                y_hat = self.feedforward(input)
                if target[y_hat] != 1:
                    loss += 1
                self.backpropagation(target, input)
            # print("Loss is ", loss / len(self.train_x) * 100, '%')

    def predict(self, test_x):
        results = []
        for example in test_x:
            example = example.reshape(example.shape[0], 1)
            results.append(str(self.feedforward(example)))
        return results


def receive_data(train_x, train_y, test_x):
    train_x = np.loadtxt(train_x)
    train_y = np.loadtxt(train_y)
    test_x = np.loadtxt(test_x)
    return train_x, train_y, test_x


def normalize_data(train_x, train_y):
    def normalize(train_x):
        train_x = train_x / 255
        return train_x

    def one_hot(train_y):
        y_vectors = np.zeros((train_y.shape[0], 10))
        for index in range(train_y.shape[0]):
            y = int(train_y[index])
            y_vectors[index][y] = 1
        return y_vectors

    return normalize(train_x), one_hot(train_y)


def validate(data_x, data_y, epochs, learning_rate, num_of_layers, k=5, normalize_weights=True):
    accuracy_per_fold = []
    len_of_validation = math.ceil(len(data_x) / k)
    print("Started validating network...")
    for _ in range(k):
        print("In fold ", _+1, f" out of {k}")
        random_state = np.random.get_state()
        np.random.shuffle(data_x)
        np.random.set_state(random_state)
        np.random.shuffle(data_y)
        validation_x = []
        validation_y = []
        for index_to_validation in range(len_of_validation):
            validation_x.append(data_x[index_to_validation])
            validation_y.append(data_y[index_to_validation])
        trainning_x = []
        trainning_y = []
        for index_to_trainning in range(len_of_validation, len(data_x)):
            trainning_x.append(data_x[index_to_trainning])
            trainning_y.append(data_y[index_to_trainning])
        newtwork = NeuralNetwork(train_x=trainning_x, train_y=trainning_y, epochs=epochs, learning_rate=learning_rate,
                                 num_of_layers=num_of_layers, normalize_weights=normalize_weights)
        newtwork.train()
        network_predictions = newtwork.predict(validation_x)
        for index, one_hot in enumerate(validation_y):
            for _, bit in enumerate(one_hot):
                if bit == 1:
                    validation_y[index] = _
                    break
        successes = sum(1 for i, j in zip(validation_y, network_predictions) if i == int(j))
        accuracy_per_fold.append((successes / len_of_validation) * 100)
    return sum(accuracy_per_fold) / len(accuracy_per_fold)


if __name__ == '__main__':
    # print(datetime.now())
    train_x, train_y, test_x = receive_data(sys.argv[1], sys.argv[2], sys.argv[3])
    train_x, train_y = normalize_data(train_x, train_y)
    network = NeuralNetwork(epochs=30, num_of_layers=3, learning_rate=0.01, train_x=train_x, train_y=train_y,
                            normalize_weights=True)
    network.train()
    output_file = open('test_y', 'w')
    test_y = network.predict(test_x)
    for y in test_y:
        output_file.write(y + '\n')
    output_file.close()
    # print(datetime.now())

