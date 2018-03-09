# -*- coding: utf-8 -*-
#初始化大权重 解决vanishing gradient problem
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import i_network2
net = i_network2.Network([784, 30, 10], cost = i_network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 30, 10, 0.5, evaluation_data = test_data, 
        monitor_evaluation_accuracy = True)

#加入lmbda,正则化L2_decay
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import i_network2
net = i_network2.Network([784, 30, 10], cost = i_network2.CrossEntropyCost)
net.SGD(training_data, 30, 10, 0.5, 5.0, evaluation_data = validation_data,
        monitor_evaluation_accuracy = True, monitor_evaluation_cost = True,
        monitor_training_accuracy = True, monitor_training_cost = True)
        