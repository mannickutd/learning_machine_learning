#!/usr/bin/env python
import numpy as np
from random import seed
from random import random
from math import exp


def initialize_network(n_inputs, n_hidden, n_outputs):
    network = []
    l1_weights = 2*np.random.random((n_inputs, n_hidden)) - 1
    l2_weights = 2*np.random.random((n_hidden, n_outputs)) - 1
    network.append({'weights': l1_weights, 'output': None, 'delta': None})
    network.append({'weights': l2_weights, 'output': None, 'delta': None})
    return network


def sigmoid(x):
    # Sigmoid
    return 1.0 / (1.0 + np.exp(-x))


def deriv(output):
    return output * (1.0 - output)


def forward_propagate(network, row):
    for i, net in enumerate(network):
        if i == 0:
            net['output'] = sigmoid(np.dot(row, net['weights']))
        else:
            net['output'] = sigmoid(np.dot(network[i-1]['output'], net['weights']))
    return network[-1]['output']


def backward_propagate_error(network, row, expected, l_rate):
    # Backpropagate
    for i, net in enumerate(reversed(network)):
        if i == 0:
            error = np.array(expected) - net['output'] 
            net['delta'] = error * deriv(net['output'])
        else:
            error = np.dot(network[-i]['delta'], network[-i]['weights'].T)
            net['delta'] = error * deriv(net['output'])
    # Update weights
    for i, net in enumerate(network):
        if i == 0:
            net['weights'] += np.dot(row.T, net['delta'])
        else:
            net['weights'] += np.dot(network[i - 1]['output'].T, net['delta'])


def train_network(network, train, test, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for i, row in enumerate(train):
            n_row = np.array([row])
            # Forward propagate
            outputs = forward_propagate(network, n_row)
            # Calculate expected vector
            expected = [0 for i in range(n_outputs)]
            expected[test[i]] = 1
            # Calculate the total error of the dataset
            sum_error += sum([(expected[i]-outputs[0][i])**2 for i in range(len(expected))])
            # Back propagate the errors
            backward_propagate_error(network, n_row, [expected], l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


def predict(network, row):
    outputs = forward_propagate(network, row)
    # Return the highest output
    return np.argmax(outputs)
