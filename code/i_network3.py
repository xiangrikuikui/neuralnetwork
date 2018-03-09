# -*- coding: utf-8 -*-

# Standard library
import cPickle
import gzip

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import pool


# Activation functions for neurons
def linear(z): return z
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

#### Constants
GPU = True
if GPU:
    print "Trying to run under a GPU. If this is not desired, then modify " +\
          "network3.py to set the GPU flag to False."
    try:
        theano.config.device = 'gpu'
    except:
        pass  # it's already set
    theano.config.floatX = 'float32'
else:
    print "Running with a CPU. If this is not desired, then the modify " +\
          "network3.py to set the GPU flag to True."



#### Load the MNIST data
def load_data_shared(filename="../data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    def shared(data):
        """Place the data into shared variables. This allows Theano to copy
        the data to the GPU, if one is available."""
        shared_x = theano.shared(
                np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
                np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]
    



#### Main class used to construct and train networks
class Network(object):
    
    def __init__(self, layers, mini_batch_size):
        """Take a list of 'layers', describing the network architecture, and
        a value for the 'mini_batch_size' to be used during training by
        stochastic gradient descent."""
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        init_layer = self.layers[0]
        """这里调用第一层的set_inpt函数。传入的inpt和inpt_dropout都是self.x，
        因为不论是训练还是测试，第一层的都是x。"""
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in xrange(1, len(self.layers)):
            prev_layer, layer = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                    prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout
        
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, 
            validation_data, test_data, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data
        
        #Compute number of minibatches for training, validation and testing
        num_training_batches = size(training_data)/mini_batch_size
        num_validation_batches = size(validation_data)/mini_batch_size
        num_test_batches = size(test_data)/mini_batch_size
        
        # define the (regularized) cost function, symbolic gradients, and updates
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self) +\
               0.5*lmbda*l2_norm_squared/num_training_batches
        #使用theano自带的梯度函数，让cost对w和b求偏导
        grads = T.grad(cost, self.params)
        updates = [(param, param-eta*grad) 
                   for param, grad in zip(self.params, grads)]
        
        
        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches
        i = T.lscalar()  # mini-batch index
        """定义训练函数，train_mb函数的输入是i，输出是cost，batch的x和y通过givens制定"""
        train_mb = theano.function(
                [i], cost, updates=updates,
                givens={
                    self.x:
                    training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                    self.y:
                    training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
                })
        #定义validation和测试函数
        validation_mb_accuracy = theano.function(
                [i], self.layers[-1].accuracy(self.y),
                givens={
                    self.x:
                    validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                    self.y:
                    validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
                })
        """输出是最后一层的accuracy self.layers[-1].accuracy(self.y)。accuracy使用的
        是最后一层的output，从而每一层都是用计算图的inpt->output路径。"""
        test_mb_accuracy = theano.function(
                [i], self.layers[-1].accuracy(self.y),
                givens={
                    self.x:
                    test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                    self.y:
                    test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
                })
        """输出是最后一层的y_out，也就是softmax的argmax(output),即输出概率最大的分类标签"""
        self.test_mb_predictions = theano.function(
                [i], self.layers[-1].y_out,
                givens={
                    self.x:
                    test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
                })
    
        
        #Do the actual training
        best_validation_accuracy = 0.0
        for epoch in xrange(epochs):
            for minibatch_index in xrange(num_training_batches):
                iteration = num_training_batches*epoch+minibatch_index
                #每1000次迭代显示一次
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                cost_ij = train_mb(minibatch_index)
                #一个epoch的最后一次迭代完成后，计算精度求精度
                if (iteration+1) % num_training_batches == 0:
                    validation_accuracy = np.mean(
                        [validation_mb_accuracy(j) for j in xrange(num_validation_batches)])
                    print("Epoch {0}: validaton accuracy {1:.2%}".format(
                            epoch, validation_accuracy))
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean(
                                    [test_mb_accuracy(j) for j in xrange(num_test_batches)])
                            print("The corresponding test accuracy is {0:.2%}".format(
                                    test_accuracy))                          
        print("Finished training network.")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
                best_validation_accuracy, best_iteration))
        print("Corresponding test sccuracy of {0:.2%}".format(test_accuracy))
        
        

#### Define layer types
class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer. A more sophisticated implementation would seperate the two,
    but for our purposes we'll always use them together, and it simplifies
    the code, so it makes sense to combine them."""
    
    def __init__(self, filter_shape, image_shape, poolsize=(2,2),
                 activation_fn=sigmoid):
        """'filter_shape' is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.filter_shape 
        (num_filter, input_feature_map, filter_width, filter_height) 这个参数是filter
        的参数，第一个是这一层的filter的个数，第二个是输入特征映射的个数，第三个是
        filter的width，第四个是filter的height
        'image_shape' is a tuple of length 4, whose tntries are the 
        mini-batch size, the number of input feature maps, the image
        height, and the image width.
        'poolsize' is a tuple of length 2, whose entries are the y and 
        x pooling sizes."""
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn = activation_fn
        # initialize weights and biases
        #prod：乘积
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        #loc均值，scale标准差
        self.w = theano.shared(
                np.asarray(
                    np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                    dtype=theano.config.floatX),
                borrow=True)
        self.b = theano.shared(
                np.asarray(
                    np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                    dtype=theano.config.floatX),
                borrow=True)
        self.params = [self.w, self.b]
    
    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        """使用theano提供的conv2d op计算卷积"""
        #conv_out = conv.conv2d(
        conv_out = conv2d(
                input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
                input_shape=self.image_shape)
        """使用theano提供的pool_2d定义pooled_out"""
        pooled_out = pool.pool_2d(
                input=conv_out, ws=self.poolsize, ignore_border=True)
        """值得注意的是dimshuffle函数，pooled_out是(batch_size, num_filter, out_width, out_height)，
        b是num_filter的向量。我们需要通过broadcasting让所有的pooled_out都加上一个bias，所以我们
        需要用dimshuffle函数把b变成(1,num_filter, 1, 1)的tensor。dimshuffle的参数’x’表示增加一个
        维度，数字0表示原来这个tensor的第0维。 dimshuffle(‘x’, 0, ‘x’, ‘x’))的意思就是在原来这个
        vector的前面插入一个维度，后面插入两个维度，所以变成了(1,num_filter, 1, 1)的tensor。"""
        self.output = self.activation_fn(


                pooled_out + self.b.dimshuffle('x',0,'x','x'))
        """卷积层没有dropout，所以output和output_dropout是同一个符号变量"""
        self.output_dropout = self.output  #no dropout in the convolutional layers
        

class FullyConnectedLayer(object):
    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
                np.asarray(
                        np.random.normal(
                                loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                        dtype = theano.config.floatX), 
                name='w', borrow=True)                             
        self.b = theano.shared(
                np.asarray(
                        np.random.normal(
                                loc=0.0, scale=1.0, size=(n_out,)),        
                        dtype = theano.config.floatX), 
                name='b', borrow=True)
        self.params = [self.w, self.b]
        
    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        """首先把input reshape成(batch_size, n_in)，为什么要reshape呢？
        因为我们在CNN里通常在最后一个卷积pooling层后加一个全连接层，
        而CNN的输出是4维的tensor(batch_size, num_filter, width, height)，
        我们需要把它reshape成(batch_size, num_filter * width * height)。"""
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
                (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        """这个计算最终的输出，也就是当这一层作为最后一层的时候输出的分类结果。
        ConvPoolLayer是没有实现y_out的计算的，因为我们不会把卷积作为网络的输出层，
        但是全连接层是有可能作为输出的，所以通过argmax来选择最大的那一个作为输出。
        SoftmaxLayer是经常作为输出的，所以也实现了y_out。"""
        self.y_out = T.argmax(self.output, axis=1)
        """inpt_dropout 先reshape，然后加一个dropout的op，
            这个op就是随机的把一些神经元的输出设置成0"""
        self.inpt_dropout = dropout_layer(
                inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
                T.dot(self.inpt_dropout, self.w) + self.b)
        
    def accuracy(self, y):
        """Return the accuracy for the mini-batch."""
        return T.mean(T.eq(y, self.y_out))
    
class SoftmaxLayer(object):
    
    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        """当我们实现softmax层时，我们没有讨论怎么初始化weights和biases。
        之前我们讨论过sigmoid层怎么初始化参数，但是那些方法不见得就适合softmax层。
        这里直接初始化成0了。这看起来很随意，不过在实践中发现没有太大问题。"""
        self.w = theano.shared(
                np.zeros((n_in, n_out), dtype=theano.config.floatX),
                name='w', borrow=True)
        self.b = theano.shared(
                np.zeros((n_out,), dtype=theano.config.floatX),
                name='b', borrow=True)
        self.params = [self.w, self.b]
        
    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
                inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)
        
    def cost(self, net):
        """Return the log-likelihood cost."""
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])
    
    def accuracy(self, y):
        return T.mean(T.eq(y, self.y_out))
    
    
### Miscellanea
def size(data):
    """Return the size of the dataset 'data'."""
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
            np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)
