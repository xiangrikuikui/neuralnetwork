import mnist_loader
import i_network1


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print("training data")
print(type(training_data))
print(len(training_data))
print(training_data[0][0].shape)
print(training_data[0][1].shape)
#print(training_data[0])

print("validation data")
print(len(validation_data))

print("test data")
print(len(test_data))

net = i_network1.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# net = network.Network([784, 100, 10])
# net.SGD(training_data, 30, 10, 0.001, test_data=test_data)

# net = network.Network([784, 100, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


# net = network.Network([784, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
