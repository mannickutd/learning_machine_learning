#!/usr/bin/env python
from random import seed
from random import random
from math import exp


def initialize_network(n_inputs, n_hidden, n_outputs):
    '''
    Keyword arguments
    n_inputs -- The number of inputs
    n_hidden -- The number of neurons in the hidden layer
    n_outputs -- Number of output classes
    '''
    network = list()
    # Its usual to initialise the weights to a random number between 0-1
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation


def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


def transfer_derivative(output):
    return output * (1.0 - output)


def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


def backward_propagate_error(network, expected):
    # Work back through each layer of the network
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            # For each neuron in the layer
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            # For the output layer
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            # Update the weights
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            # Update the bias
            neuron['weights'][-1] += l_rate * neuron['delta']


def train_network(network, train, test, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for i, row in enumerate(train):
            # Forward propagate
            outputs = forward_propagate(network, row)
            # Calculate expected vector
            expected = [0 for i in range(n_outputs)]
            expected[test[i]] = 1
            # Calculate the total error of the dataset
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            # Back propagate the errors
            backward_propagate_error(network, expected)
            # Updat the weights from the error
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


def predict(network, row):
    outputs = forward_propagate(network, row)
    # Return the highest output
    return outputs.index(max(outputs))
