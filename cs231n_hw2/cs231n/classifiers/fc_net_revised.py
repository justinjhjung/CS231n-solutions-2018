from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
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
        
        # Initialize weights and biases
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        
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
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N = X.shape[0]
        
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        
        # get score
        X_re = X.reshape(N,-1)
        H = X_re.dot(W1) + b1
        relu = np.maximum(0, H)
        scores = relu.dot(W2) + b2 # Shape : (N, C)
        
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
        
        ##################### LOSS #####################
        
        # get softmax classifier loss
        scores -= np.amax(scores, axis=1)[:,None]
        scores_y = scores[np.arange(N),y][:,None]
        exp_scores = np.exp(scores)
        exp_sum = np.sum(exp_scores, axis=1)[:,None]
        loss = np.sum(-scores_y + np.log(exp_sum)) / N

        # add l2 regularization for W1 and W2
        loss += 0.5 * self.reg * np.sum(W1*W1)
        loss += 0.5 * self.reg * np.sum(W2*W2)
        
        ################## GRADIENTS ##################
        
        # initial setting for gradients 
        dW1 = np.zeros_like(W1) # Shape : (D,H)
        dW2 = np.zeros_like(W2) # Shape : (H,C)
        
        # get gradient for W2
        temp1 = exp_scores/exp_sum  # Shape : (N,C)
        temp1[np.arange(N),y] -= 1
        dW2 = relu.T.dot(temp1) / N + self.reg * W2 # Shape : (H,C)
        grads['W2'] = dW2 
        
        # get gradient for b2
        db2 = np.sum(temp1, axis=0) / N # Shape : (1,C) 
        grads['b2'] = db2 
        
        # get gradients for W1
        drelu = temp1.dot(W2.T) # Shape : (N,H)
        dH = drelu * (H > 0) # Shape : (N,H)
        dW1 = X_re.T.dot(dH) / N + self.reg * W1 # Shape : (D,H)
        grads['W1'] = dW1
        
        # get gradient for b1
        db1 = np.sum(dH, axis=0) / N # Shape : (1,H)
        grads['b1'] = db1
        
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
        self.weight_scale = weight_scale
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

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
        
        # Initialize weights and biases
        self.weight_names = []
        self.bias_names = []
        
        for i in range(self.num_layers):
            # getting dimensions
            if i != 0:
                input_dim = hidden_dims[i-1]
                if i != self.num_layers-1:
                    output_dim = hidden_dims[i]
                else:
                    output_dim = num_classes
            else:
                output_dim = hidden_dims[i]
                
            weight_name = "W%d" % (i+1) 
            bias_name = "b%d" % (i+1)
            self.weight_names.append(weight_name)
            self.bias_names.append(bias_name)
            
            # putting inital weights and biases to the self.params
            self.params[weight_name] = self.weight_scale * np.random.randn(input_dim, output_dim)
            self.params[bias_name] = np.zeros(output_dim)
            
            # Initialize gammas and betas for batchnorm 
            if self.normalization == "batchnorm":
                
                # generating names for gammas and betas
                self.gamma_names = []
                self.beta_names = []
                
                scale_name = "gamma%d" % (i+1)
                shift_name = "beta%d" % (i+1)
                self.gamma_names.append(scale_name)
                self.beta_names.append(shift_name)
                
                # putting initial gammas and betas to the self.params
                print("hidden_dims", hidden_dims)
                self.params[scale_name] = np.ones(output_dim)
                self.params[shift_name] = np.zeros(output_dim)
                
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
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

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
        
        N = X.shape[0]
        
        # get score + caching neccessary components to all the layers
        cache = {'weights':[],'biases':[],'logits':[],'outputs':[]}
        
        in_ = X.reshape(N,-1)
        out_ = None
        cache['outputs'].append(in_) # not to save input, I also considered the first input as an output.
        
        for i in range(self.num_layers):
            weight = self.params[self.weight_names[i]]
            bias = self.params[self.bias_names[i]]
            logit_ = in_.dot(weight) + bias
            
            if self.normalization == 'batchnorm':
                N, D = logit_.shape
                x = logit_.copy()  # x is different from the original input X
                
                # get gamma and beta from params
                gamma = self.params[self.gamma_names[i]]
                beta = self.params[self.beta_names[i]]
                
                # get/set values for calculating batch norm test mode output
                running_mean = self.bn_params[i].get('running_mean', np.zeros(D, dtype=x.dtype))
                running_var = self.bn_params[i].get('running_var', np.zeros(D, dtype=x.dtype))
                eps = self.bn_params[i].get('eps', 1e-5)
                momentum = self.bn_params[i].get('momentum', 0.9)
                
                # computation when mode is train
                if self.bn_params[i]['mode'] == 'train':
                    x = logit_.copy()  # x is different from the original input X
                    
                    # get sample mean and variance
                    sample_mean = np.mean(x, axis=0)
                    sample_var = np.var(x, axis=0)
                    
                    # get running mean and variance
                    running_mean = momentum * running_mean + (1 - momentum) * sample_mean # Shape : (D,)
                    running_var = momentum * running_var + (1 - momentum) * sample_var    # Shape : (D,)
                    
                    # normalize and scale/shift with sample mean and variance
                    xhat = (x - sample_mean) / np.sqrt(sample_var + eps)
                    print("gamma", gamma.shape)
                    print("xhat", xhat.shape)
                    print("beta", beta.shape)
                    logit_ = gamma * xhat + beta
                    
                    # update params
                    cache['logit_oris'] = x
                    cache['xhat'] = xhat
                    cache['gamma'] = gamma
                    cache['beta'] = beta
                    cache['sample_mean'] = sample_mean
                    cache['sample_var'] = sample_var
                    
                    self.bn_params[i]['eps'] = eps
                    self.bn_params[i]['momentum'] = momentum
                    self.bn_params[i]['running_mean'] = running_mean
                    self.bn_params[i]['running_var'] = running_var
                    
                else:
                    # normalize and scale/shift with running_mean and running_var
                    xhat = (x - running_mean) / np.sqrt(running_var + eps)
                    logit_ = gamma * xhat + beta
                
            if i != self.num_layers-1:
                out_ = np.maximum(0, logit_)
            else:
                scores = logit_
                cache['outputs'].append(scores)
            
            # cache neccessary components of the layer
            cache['weights'].append(weight) 
            cache['biases'].append(bias)     
            cache['logits'].append(logit_)
            cache['outputs'].append(out_)   
            
            # update in_ with the output from the previous layer
            in_ = out_
            
        
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
        
        ##################### LOSS #####################
        
        # get softmax classifier loss
        scores -= np.amax(scores, axis=1)[:,None]
        scores_y = scores[np.arange(N),y][:,None]
        exp_scores = np.exp(scores)
        exp_sum = np.sum(exp_scores, axis=1)[:,None]
        loss = np.sum(-scores_y + np.log(exp_sum)) / N

        # add l2 regularization for weights
        for i in range(self.num_layers):
            weight = cache['weights'][i]
            loss += 0.5 * self.reg * np.sum(weight*weight)
        
        ################## GRADIENTS ##################
        
        # softmax gradient to the output of the last layer
        dout = exp_scores/exp_sum
        dout[np.arange(N),y] -= 1
        
        for i in reversed(range(self.num_layers)):
            
            # generate gradient names
            weight_name = "W%d" % (i+1) 
            bias_name = "b%d" % (i+1)
            
            # get cached items (let's assume now we are at the 'i'th layer)
            weight = cache['weights'][i]         # Shape : (H_<i-1>, H_<i>)
            logit = cache['logits'][i]           # Shape : (N, H_<i>)
            out_prev = cache['outputs'][i]       # Shape : (N, H_<i-1>)

            # compute gradients
            if i != self.num_layers-1:
                weight_aft = cache['weights'][i+1] # Shape : (H_<i>, H_<i+1>)
                dout = dout.dot(weight_aft.T)      # Shape : (N, H_<i>)
                grads[weight_name] = out_prev.T.dot(dout * (logit > 0)) / N + self.reg * weight # Shape : (H_<i-1>, H_<i>)
                grads[bias_name] = np.sum(dout, axis=0) / N
                
            else:
                grads[weight_name] = out_prev.T.dot(dout) / N + self.reg * weight # Shape : (H_<i-1>, H_<i>)
                grads[bias_name] = np.sum(dout, axis=0) / N

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
