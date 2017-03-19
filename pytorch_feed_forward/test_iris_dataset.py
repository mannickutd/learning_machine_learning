#!/usr/bin/env python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from feed_forward import (initialize_network, train_network, predict)


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
	l_rate = 0.001
	n_epochs = 100
	n_hidden = 6
	network = initialize_network(n_inputs, n_hidden, n_outputs)
	train_network(network, x_train, y_train, l_rate, n_epochs, n_outputs)
	#y_pred = [predict(network, i) for i in x_test]
	#print("Accuracy score {0}".format(accuracy_score(y_test, y_pred)))