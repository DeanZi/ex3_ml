# ex3_ml

### Neural Network for MNIST Classification
This code contains a class called NeuralNetwork, which is used for training and testing on the MNIST dataset. 

### Dependencies
* numpy
* random
* scipy

### Functions
The following functions are used in the NeuralNetwork class:

* init_parameters: Initializes the weights and biases for the neural network. The weights are initialized randomly and the biases are set to zero.
* feedforward: Performs the feedforward operation for the neural network. The input is passed through each layer of the network and the output is returned.
* backpropagation: Performs the backpropagation operation for the neural network. The gradients of the weights and biases are calculated for each layer using the chain rule of differentiation.
* train: Trains the neural network on the MNIST dataset using stochastic gradient descent.
* predict: Makes predictions using the trained neural network on a given input.

### Activation functions
The following activation functions are used in the neural network:

* sigmoid
* dsigmoid 
* relu
* drelu
* softmax

### Results

* Learning rate check on 1 layer: 0.01 --> 81.541% accuracy
* Epochs check on 1 layer: 30 -->  82.083% accuracy
* Best result was receivd on 3 layers: 94% accuracy


### Author
This code was written by *DeanZi* as part of a university programming project. Feel free to use and modify it as you like.
