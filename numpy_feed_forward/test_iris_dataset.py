#!/usr/bin/env python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#from numpy_feed_forward import (init_network, train_network, predict)


if __name__ == '__main__':
    # This data sets consists of 3 different types of irises’ (Setosa, Versicolour, and Virginica)
    # The iris data set has 4 features Sepal Length, Sepal Width, Petal Length and Petal Width.
    iris = datasets.load_iris()
    # Split the dataset into test and train.
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.20)
    # Network properties
    _, n_inputs = x_train.shape
    n_outputs = len(set([x for x in iris.target]))
    # Train parameters
    # l_rate = 0.5
    # n_epochs = 100
    # n_hidden = 6
    # network = init_network(n_inputs, n_hidden, n_outputs)
    # train_network(network, x_train, y_train, l_rate, n_epochs, n_outputs)
    # y_pred = [predict(network, i) for i in x_test]
    # print("Accuracy score {0}".format(accuracy_score(y_test, y_pred)))
    from numpy_feed_forward import Neuralnet
    activation1 = 'sigmoid'     # the input should be scaled into [ 0..1]
    net = Neuralnet([4, 6, 3], 'sigmoid')
    Y_scaled = []
    for i in y_train:
        expected = [0 for i in range(n_outputs)]
        expected[y_train[i]] = 1
        Y_scaled.append(expected)

    for i, x in enumerate(x_train):
        net.train(x, Y_scaled[i], 10)

    trained_weights = net.weights
    
    Y_scaled = []
    for i in y_test:
        expected = [0 for i in range(n_outputs)]
        expected[y_test[i]] = 1
        Y_scaled.append(expected)


    Y_pred = net.predict(np.array(Y_scaled), trained_weights)

    print(Y_pred)
