from builtins import range
from builtins import object
import numpy as np
from cs231n.layers import *  # cs231n.
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params["W1"] = np.random.randn(input_dim, hidden_dim) * weight_scale
        self.params["b1"] = np.zeros(hidden_dim)  # np.random.randn(hidden_dim)
        self.params["W2"] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params["b2"] = np.zeros(num_classes)  # np.random.randn(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        N = X.shape[0]
        D = np.prod(X[1, :].shape)  # self.params["W1"].shape[0]
        hidden_layer, cache_hidden_layer = affine_relu_forward(X, self.params["W1"], self.params["b1"])
        scores, cache_scores = affine_forward(hidden_layer, self.params["W2"], self.params["b2"])  # 未经过soft max运算
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        dhidden, dw2, db2 = affine_backward(dscores, cache_scores)
        _, dw1, db1 = affine_relu_backward(dhidden, cache_hidden_layer)
        dw2 += self.reg * self.params["W2"]
        dw1 += self.reg * self.params["W1"]
        grads = {"W1": dw1,
                 "b1": db1,
                 "W2": dw2,
                 "b2": db2
                 }
        loss += 0.5 * self.reg * (np.sum(self.params["W1"] * self.params["W1"]) + np.sum(self.params["W2"] * self.params["W2"]))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = dict()

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################

        assert type(hidden_dims) == list, "hidden_dims must be a list!"

        for i in range(self.num_layers - 1):
            self.params['W' + str(i + 1)] = np.random.normal(0, weight_scale, [input_dim, hidden_dims[i]])
            self.params['b' + str(i + 1)] = np.zeros([hidden_dims[i]])
            if self.normalization == "batchnorm" or self.normalization == "layernorm":
                self.params['beta' + str(i + 1)] = np.zeros([hidden_dims[i]])
                self.params['gamma' + str(i + 1)] = np.ones([hidden_dims[i]])
            input_dim = hidden_dims[i]  # Set the input dim of next layer to be output dim of current layer.
        # Initialise the weights and biases for final FC layer
        self.params['W' + str(self.num_layers)] = np.random.normal(0, weight_scale, [input_dim, num_classes])
        self.params['b' + str(self.num_layers)] = np.zeros([num_classes])
        # dims = [input_dim] + hidden_dims + [num_classes]
        # Ws = {"W" + str(i + 1): weight_scale * np.random.randn(dims[i], dims[i + 1]) for i in range(len(dims) - 1)}
        # b = {"b" + str(i + 1): np.zeros(dims[i + 1]) for i in range(len(dims) - 1)}
        # self.params.update(Ws)
        # self.params.update(b)
        # if self.normalization == "batchnorm" or self.normalization == "layernorm":
        #     for i in range(self.num_layers - 1):
        #         self.params['beta' + str(i + 1)] = np.zeros([hidden_dims[i]])
        #         self.params['gamma' + str(i + 1)] = np.ones([hidden_dims[i]])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for _ in range(self.num_layers - 1)]  # change i to _
        if self.normalization=='layernorm':
            self.bn_params = [{} for _ in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################

        # 重新分析结构，最复杂的莫过于：[affine--batch norm--relu] --dropout --[affine--batch norm--relu]--dropout--affine--softmax
        # 当然，这里的[]可以有很多个，定义字典：hidden_layers:存放各层运算的输入，第一个为X，其余是各层的输出，显然dropout只是对结果进行随机丢弃
        # 共有1 + L + 1, L为隐层[]的数目，1是最后一层affine的结果，不包含softmax, caches存放各层的cache,共L+1个。dp_cache: 也就是dropout的结果
        # dropout至多有L个

        hidden_layers, caches = [_ for _ in range(self.num_layers + 1)], [_ for _ in range(self.num_layers)]
        dp_caches = np.zeros(self.num_layers - 1)  # dropout layer 至多有num_layer - 1个，因为只能在隐层之后有，输出层和输入层无
        hidden_layers[0] = X
        for i in range(self.num_layers):  # iter num_layers
            W, b = self.params["W" + str(i + 1)], self.params['b' + str(i + 1)]
            if i == self.num_layers - 1:
                hidden_layers[i + 1], caches[i] = affine_forward(hidden_layers[i], W, b)  # 最后一层计算scores
            else:
                if self.normalization == "batchnorm" or self.normalization == "layernorm":
                    gamma, beta = self.params['gamma' + str(i + 1)], self.params['beta' + str(i + 1)]
                if self.normalization == "batchnorm":
                    hidden_layers[i + 1], caches[i] = affine_bn_relu_forward(hidden_layers[i], W, b, gamma, beta,
                                                                             self.bn_params[i])
                elif self.normalization == "layernorm":
                    ln_param = {"eps": 1e-5}  # 简单粗暴
                    hidden_layers[i + 1], caches[i] = affine_layernorm_relu_forward(hidden_layers[i], W, b, gamma, beta, ln_param)
                else:
                    hidden_layers[i + 1], caches[i] = affine_relu_forward(hidden_layers[i], W, b)
                if self.use_dropout:
                    hidden_layers[i + 1], dp_caches[i] = dropout_forward(hidden_layers[i + 1], self.dropout_param)

        scores = hidden_layers[-1]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        data_loss, dscores = softmax_loss(scores, y)
        weight = [np.sum(self.params["W" + str(i + 1)] * self.params["W" + str(i + 1)]) for i in range(self.num_layers - 1)]
        reg_loss = self.reg * 0.5 * np.sum(weight)
        loss = data_loss + reg_loss

        # backward
        dhidden = dscores
        dhiddens = [_ for _ in range(self.num_layers + 1)]
        dhiddens[self.num_layers] = dscores
        for i in range(self.num_layers, 0, -1):  # iter num_layers from num_layers to 1
            # define input
            dout = dhidden
            cache = caches[i - 1]

            # the last layer, just need affine backward
            if i == self.num_layers:
                dhidden, dw, db = affine_backward(dout, cache)
                dhiddens[i - 1] = dhidden
                grads["W" + str(i)], grads["b" + str(i)] = dw, db
            # other layer
            else:
                # make dropout as a kind of special operation, return to the last dhidden, not next dhidden
                if self.use_dropout:
                    dhidden = dropout_backward(dout, dp_caches[i - 1])  # 不知道为啥dp_cache,显然最后一个affine后不跟dropout
                    dhiddens[i] = dhidden
                if self.normalization == "batchnorm":
                    dhidden, dw, db, dgamma, dbeta = affine_bn_relu_backward(dout, cache)
                    grads["gamma" + str(i)], grads["beta" + str(i)] = dgamma, dbeta
                elif self.normalization == "layernorm":
                    dhidden, dw, db, dgamma, dbeta = affine_layernorm_relu_backward(dout, cache)
                    grads["gamma" + str(i)], grads["beta" + str(i)] = dgamma, dbeta
                else:
                    dhidden, dw, db = affine_relu_backward(dout, cache)
                dhiddens[i - 1] = dhidden
                grads["W" + str(i)], grads["b" + str(i)] = dw, db

            grads['W' + str(i)] += self.reg * self.params['W' + str(i)]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
