# -*- coding: utf-8 -*-

import numpy as np
import random

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
    
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))



class Network(object):
    def __init__(self, sizes):  #构造函数
        self.num_layers = len(sizes)  #有几层
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  #从第二层开始
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        """Train the neural network using mini_batch stochastic gradient descent. 
        The "training_data" is a list of tuples "(x, y)" representing the training
        inputs and the desired outputs. The other non-optional parameters are 
        self-explanatory. If "test_data" is provided then the network will be
        evaluated against the test data after each epoch, and partial progress 
        printed out. This is useful for tracking progress, but slows things down 
        substantially."""
        if test_data:
            n_test = len(test_data)
            n = len(training_data)
            for j in xrange(epochs):
                random.shuffle(training_data)  #随机打乱training_data,每次抽取mini_batch
                mini_batches = [
                        training_data[k:k+mini_batch_size]
                        for k in xrange(0, n, mini_batch_size)]
                for mini_batch in mini_batches:
                    self.update_mini_batch(mini_batch, eta)
                if test_data:
                    print "Epoch {0}: {1} / {2}".format(
                            j, self.evaluate(test_data), n_test)
                else:
                    print "Epoch {0} complete".format(j)
    
    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying gradient descent using  
        backpropagation to a single mini batch. The "mini_batch" is a list of tuples
        "(x, y)", and "eta" is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            #求b和w的偏导
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            #对所有的b偏导求和
            #nb也就是nabla_b初始化为全0
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                        for b, nb in zip(self.biases, nabla_b)]
    

    def backprop(self, x, y):
        """Return a tuple "(nabla_b, nabla_w)" representing the gradient for the cost
        function C_x. "nabla_b" and "nabla_w" are layer-by-layer lists of numpy arrays,
        similar to "self.biases" and "self.weights"."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #feedforward
        activation = x  #输入层的值，具体来讲784维向量
        activations = [x]  # list to store all activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #backward pass
        #cost_derivative(activations[-1], y)：C对a求导，即[1/2*power((a-y),2)]对a求导
        #即就是a-y
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        #cost对b和w的偏导
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        """Note that variable l in the loop below is used a little differently to 
        the notation in Chapter 2 of the book. Here, l = -1 means the last layer of 
        neurons, l = -2 is the second-last layer, and so on. It's a renumbering of 
        the scheme in the book, used here to take advantages of the fact that Python
        can use negative indices in lists."""
        #从倒数第二层往回更新
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            #反向跟新error
            #delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def evaluate(self, test_data):
        """Return the number of the test inputs for which the neural network ouptuts
        the corrtect result. Note that the neural network's output is assumed to be the 
        index of whichever neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x / \partial a for the
        output activations."""
        return (output_activations - y)
