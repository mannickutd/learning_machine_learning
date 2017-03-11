#!/usr/bin/env python
import numpy as np


import numpy as np
import math
#import matplotlib.pyplot as plt

class Neuralnet:
    def __init__(self, neurons, activation):
        self.weights = []
        self.inputs = []
        self.outputs = []
        self.errors = []
        self.rate = 0.5
        self.activation = activation    #sigmoid or tanh

        self.neurons = neurons
        self.L = len(self.neurons)      #number of layers

        eps = 0.12;    # range for uniform distribution   -eps..+eps              
        for layer in range(len(neurons)-1):
            self.weights.append(np.random.uniform(-eps,eps,size=(neurons[layer+1], neurons[layer]+1)))            


    ###################################################################################################    
    def train(self, X, Y, iter_count):

        m = X.shape[0];

        for layer in range(self.L):
            self.inputs.append(np.empty([m, self.neurons[layer]]))        
            self.errors.append(np.empty([m, self.neurons[layer]]))

            if (layer < self.L -1):
                self.outputs.append(np.empty([m, self.neurons[layer]+1]))
            else:
                self.outputs.append(np.empty([m, self.neurons[layer]]))

        #accumulate the cost function
        J_history = np.zeros([iter_count, 1])

        m = X.shape[0]

        for i in range(iter_count):

            self.feedforward(X, m)

            J = self.cost(Y, self.outputs[self.L-1], m)
            J_history[i, 0] = J

            self.backpropagate(Y, m)


        #plot the cost function to check the descent
        #plt.plot(J_history)
        #plt.show()


    ###################################################################################################    
    def cost(self, Y, H, m):     
        J = np.sum(np.sum(np.power((Y - H), 2), axis=0))/(2*m)
        return J

    ###################################################################################################
    def feedforward(self, X, m):

        self.outputs[0] = np.concatenate(  (np.ones([m, 1]),   X),   axis=1)

        for i in range(1, self.L):
            self.inputs[i] = np.dot( self.outputs[i-1], self.weights[i-1].T  )

            if (self.activation == 'sigmoid'):
                output_temp = self.sigmoid(self.inputs[i])
            elif (self.activation == 'tanh'):
                output_temp = np.tanh(self.inputs[i])


            if (i < self.L - 1):
                self.outputs[i] = np.concatenate(  (np.ones([m, 1]),   output_temp),   axis=1)
            else:
                self.outputs[i] = output_temp

    ###################################################################################################
    def backpropagate(self, Y, m):

        self.errors[self.L-1] = self.outputs[self.L-1] - Y

        for i in range(self.L - 2, 0, -1):

            if (self.activation == 'sigmoid'):
                self.errors[i] = np.dot(  self.errors[i+1],   self.weights[i][:, 1:]  ) *  self.sigmoid_prime(self.inputs[i])
            elif (self.activation == 'tanh'):
                self.errors[i] = np.dot(  self.errors[i+1],   self.weights[i][:, 1:]  ) *  (1 - self.outputs[i][:, 1:]*self.outputs[i][:, 1:])

        for i in range(0, self.L-1):
            grad = np.dot(self.errors[i+1].T, self.outputs[i]) / m
            self.weights[i] = self.weights[i] - self.rate*grad

    ###################################################################################################
    def sigmoid(self, z):
        s = 1.0/(1.0 + np.exp(-z))
        return s

    ###################################################################################################
    def sigmoid_prime(self, z):
        s = self.sigmoid(z)*(1 - self.sigmoid(z))
        return s    

    ###################################################################################################
    def predict(self, X, weights):

        m = X.shape[0];

        self.inputs = []
        self.outputs = []
        self.weights = weights

        for layer in range(self.L):
            self.inputs.append(np.empty([m, self.neurons[layer]]))        

            if (layer < self.L -1):
                self.outputs.append(np.empty([m, self.neurons[layer]+1]))
            else:
                self.outputs.append(np.empty([m, self.neurons[layer]]))

        self.feedforward(X, m)

        return self.outputs[self.L-1]



# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))


# def deriv(x):
#     return x * (1.0 - x)


# def feed_forward(network, batch, train=False):
#     inputs = batch
#     for l in network:
#         inputs = sigmoid(np.dot(inputs, l['weights']))
#         if train:
#             l['layer'] = inputs
#     return inputs


# def back_propagate(network, expected):
#     for i, l in enumerate(reversed(network)):
#         if i == 0:
#             l['delta'] = (expected - l['layer']) * deriv(l['layer'])
#         else:
#             l['delta'] = prev_l['delta'].dot(prev_l['weights'].T) * deriv(l['layer'])
#         prev_l = l


# def update_weights(network, batch, l_rate):
#     inputs = batch
#     for i, l in enumerate(network):
#         l['weights'] += l_rate * np.dot(inputs, l['delta'])
#         inputs = network[i]['layer']


# def init_network(input_size, hidden_size, output_size):
#     # Initialise a very simple 1 hidden layer neural network
#     return [{"weights": (2 * np.random.random((input_size, hidden_size)) - 1), "layer": None, 'delta': None},
#             {"weights": (2 * np.random.random((hidden_size, output_size)) - 1), "layer": None, 'delta': None}]


# def train_network(network, train, test, l_rate, n_epoch, n_outputs):
#     for epoch in range(n_epoch):
#         sum_error = 0
#         for i, row in enumerate(train):
#             outputs = feed_forward(network, row, train=True)
#             # Calculate expected vector
#             expected = [0 for i in range(n_outputs)]
#             expected[test[i]] = 1
#             # Calculate the total error of the dataset
#             sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
#             # Back propagate the errors
#             back_propagate(network, expected)
#             # Update the weights
#             update_weights(network, row, l_rate)
#         print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


# def predict(network, row):
#     outputs = feed_forward(network, row)
#     # Return the highest output
#     return outputs.index(max(outputs))
