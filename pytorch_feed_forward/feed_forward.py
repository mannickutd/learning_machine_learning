#!/usr/bin/env python
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as onn
import torch.nn.functional as fnn
import torch.autograd as ann


class Net(nn.Module):

    def __init__(self, n_inputs, n_hidden, n_outputs):
        super(Net, self).__init__()
        self.h_layer = nn.Linear(n_inputs, n_hidden)
        self.o_layer = nn.Linear(n_hidden, n_outputs)
        self.loss_function = nn.MSELoss()
        self.optimizer = onn.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        h_output = fnn.sigmoid(self.h_layer(x))
        o_output = fnn.sigmoid(self.o_layer(h_output))
        return o_output


def initialize_network(n_inputs, n_hidden, n_outputs):
    return Net(n_inputs, n_hidden, n_outputs)


def train_network(network, train, test, l_rate, n_epoch, n_outputs):
    network.optimizer = onn.Adam(network.parameters(), lr=l_rate)
    for epoch in range(n_epoch):
        for i, x in enumerate(train):
            network.optimizer.zero_grad()
            input_tensor = ann.Variable(torch.Tensor([x]))
            expected = [0 for i in range(n_outputs)]
            expected[test[i]] = 1
            expected_tensor = ann.Variable(torch.Tensor(expected))
            output = network.forward(input_tensor)
            loss = network.loss_function(output, expected_tensor)
            loss.backward()
            network.optimizer.step()
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, loss.data[0]))


def predict(network, x):
    network.eval()
    input_tensor = ann.Variable(torch.Tensor([x]))
    output = network.forward(input_tensor)
    return np.argmax(output.data.numpy())
