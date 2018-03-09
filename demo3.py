import i_network3
from i_network3 import Network
from i_network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = i_network3.load_data_shared()
mini_batch_size = 10
net = Network([ConvPoolLayer(image_shape = (mini_batch_size, 1, 28, 28),
               filter_shape = (20, 1, 5, 5), poolsize = (2, 2)),
               FullyConnectedLayer(n_in = 20*12*12, n_out = 100),
               SoftmaxLayer(n_in = 100, n_out = 10)], mini_batch_size)
net.SGD(training_data, 60,  mini_batch_size, 0.1, validation_data, test_data)
#准确率98.78

#再加入一层convolution
import i_network3
from i_network3 import Network
from i_network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = i_network3.load_data_shared()
mini_batch_size = 10
net = Network([ConvPoolLayer(image_shape = (mini_batch_size, 1, 28, 28),
               filter_shape = (20, 1, 5, 5), poolsize = (2, 2)),
               ConvPoolLayer(image_shape = (mini_batch_size, 20, 12, 12),
               filter_shape = (40, 20, 5, 5), poolsize = (2, 2)),
               FullyConnectedLayer(n_in = 40*4*4, n_out = 100),
               SoftmaxLayer(n_in = 100, n_out = 10)], mini_batch_size)
net.SGD(training_data, 60,  mini_batch_size, 0.1, validation_data, test_data)
#准确率99.06

#采用ReLU函数代替sigmoid  
import i_network3
from i_network3 import Network
from i_network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
from i_network3 import ReLU
training_data, validation_data, test_data = i_network3.load_data_shared()
mini_batch_size = 10
net = Network([ConvPoolLayer(image_shape = (mini_batch_size, 1, 28, 28),
               filter_shape = (20, 1, 5, 5), poolsize = (2, 2),
               activation_fn = ReLU),
               ConvPoolLayer(image_shape = (mini_batch_size, 20, 12, 12),
               filter_shape = (40, 20, 5, 5), poolsize = (2, 2),
               activation_fn = ReLU),
               FullyConnectedLayer(n_in = 40*4*4, n_out = 100, 
               activation_fn = ReLU),
               SoftmaxLayer(n_in = 100, n_out = 10)], mini_batch_size)
net.SGD(training_data, 60,  mini_batch_size, 0.03, validation_data, test_data, lmbda = 0.1)
#准确率99.23

#增大训练集：每个图像向上下左右移动一个像素
$python expand_mnist.py
import i_network3
from i_network3 import Network
from i_network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
from i_network3 import ReLU
mini_batch_size = 10
expanded_training_data, validation_data, test_data = i_network3.load_data_shared("../data/mnist_expanded.pkl.gz")
net = Network([ConvPoolLayer(image_shape = (mini_batch_size, 1, 28, 28),
               filter_shape = (20, 1, 5, 5), poolsize = (2, 2),
               activation_fn = ReLU),
               ConvPoolLayer(image_shape = (mini_batch_size, 20, 12, 12),
               filter_shape = (40, 20, 5, 5), poolsize = (2, 2),
               activation_fn = ReLU),
               FullyConnectedLayer(n_in = 40*4*4, n_out = 100, 
               activation_fn = ReLU),
               SoftmaxLayer(n_in = 100, n_out = 10)], mini_batch_size)
net.SGD(expanded_training_data, 60,  mini_batch_size, 0.03, validation_data, test_data, lmbda = 0.1)
#准确率99.37


#加入一个100个神经元的隐藏层在fully-connected层：
$python expand_mnist.py
import i_network3
from i_network3 import Network
from i_network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
from i_network3 import ReLU
mini_batch_size = 10
expanded_training_data, validation_data, test_data = i_network3.load_data_shared("../data/mnist_expanded.pkl.gz")
net = Network([ConvPoolLayer(image_shape = (mini_batch_size, 1, 28, 28),
               filter_shape = (20, 1, 5, 5), poolsize = (2, 2),
               activation_fn = ReLU),
               ConvPoolLayer(image_shape = (mini_batch_size, 20, 12, 12),
               filter_shape = (40, 20, 5, 5), poolsize = (2, 2),
               activation_fn = ReLU),
               FullyConnectedLayer(n_in = 40*4*4, n_out = 100, 
               activation_fn = ReLU),
               FullyConnectedLayer(n_in = 100, n_out = 100, 
               activation_fn = ReLU),
               SoftmaxLayer(n_in = 100, n_out = 10)], mini_batch_size)
net.SGD(expanded_training_data, 60,  mini_batch_size, 0.03, validation_data, test_data, lmbda = 0.1)
#准确率99.43，并没有大的提高，有可能overfit


#加上dropout到最后一个fully-connected层
$python expand_mnist.py
import i_network3
from i_network3 import Network
from i_network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
from i_network3 import ReLU
mini_batch_size = 10
expanded_training_data, validation_data, test_data = i_network3.load_data_shared("../data/mnist_expanded.pkl.gz")
net = Network([ConvPoolLayer(image_shape = (mini_batch_size, 1, 28, 28),
               filter_shape = (20, 1, 5, 5), poolsize = (2, 2),
               activation_fn = ReLU),
               ConvPoolLayer(image_shape = (mini_batch_size, 20, 12, 12),
               filter_shape = (40, 20, 5, 5), poolsize = (2, 2),
               activation_fn = ReLU),
               FullyConnectedLayer(n_in = 40*4*4, n_out = 1000, 
               activation_fn = ReLU, p_dropout = 0.5),
               FullyConnectedLayer(n_in = 1000, n_out = 100, 
               activation_fn = ReLU, p_dropout = 0.5),
               SoftmaxLayer(n_in = 100, n_out = 10, p_dropout = 0.5)], mini_batch_size)
net.SGD(expanded_training_data, 40,  mini_batch_size, 0.03, validation_data, test_data, lmbda = 0.1)
#准确率99.60 显著提高
#epochs:减少到了40
#隐藏层有1000个神经元


#ensemble of network：训练多个神经网络，投票决定结果，有时会提高