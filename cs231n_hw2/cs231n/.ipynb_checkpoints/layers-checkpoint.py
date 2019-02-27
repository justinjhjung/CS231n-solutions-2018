from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    
    # number of sample
    N = x.shape[0]
    
    # compute forward pass
    x_rs = x.reshape(N, -1)# Shape : (N, D)
    out = x_rs.dot(w) + b  # Shape : (N, M)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    
    # num_sample 
    N = x.shape[0]
    
    # Reshape x
    x_rs= x.reshape(N, -1)

    # get gradients
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x_rs.T.dot(dout)
    db = np.sum(dout, axis=0)
    
 ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    
    out = np.maximum(0, x)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    
    dx = dout * (x > 0)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        
        cache = {}
        
        # get sample mean and variance
        sample_mean = np.mean(x, axis=0) # Shape : (D,)
        sample_var = np.var(x, axis=0)   # Shape : (D,)
        
        # get running mean and variance
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean # Shape : (D,)
        running_var = momentum * running_var + (1 - momentum) * sample_var    # Shape : (D,)
        
        # normalize and scale/shift with sample mean and variance
        x_norm = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * x_norm + beta
        
        # caching neccessary variables for backward pass
        cache['x'] = x
        cache['xhat'] = x_norm
        cache['gamma'] = gamma
        cache['eps'] = eps
        cache['sample_mean'] = sample_mean
        cache['sample_var'] = sample_var
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        
        # normalize and scale/shift
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta
        
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    
    # get cached items
    x, xhat, gamma, eps, sample_mean, sample_var = \
    cache['x'], cache['xhat'], cache['gamma'], cache['eps'], \
    cache['sample_mean'], cache['sample_var']
    N, _ = x.shape
    
    # get gradient for dgamma and dbeta
    dgamma = np.sum(xhat * dout, axis=0)
    dbeta = np.sum(dout, axis=0)
    
    # get gradient for dx
    dxhat = dout * gamma # Shape : (N, D)
    dvar = np.sum(dxhat * (x - sample_mean), axis=0) * -1/2 * ((sample_var + eps)**(-3/2)) # Shape : (D,)
    dmu = np.sum(dxhat * -1/np.sqrt(sample_var + eps), axis=0) + dvar * -2 * np.mean(x - sample_mean, axis=0) # Shape : (D,)
    dx = dxhat * 1/np.sqrt(sample_var + eps) + dvar * 2 * (x - sample_mean)/N + (dmu/N)[None,:] # Shape : (N, D)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    
    # get cached items
    x, xhat, gamma, eps, sample_mean, sample_var = \
    cache['x'], cache['xhat'], cache['gamma'], cache['eps'], \
    cache['sample_mean'], cache['sample_var']
    
    # get gradient for dgamma and dbeta
    dgamma = np.sum(xhat * dout, axis=0)
    dbeta = np.sum(dout, axis=0)
    
    # get gradient for dx
    invsig = (sample_var + eps)**(-0.5)
    x_centered = x - sample_mean
    dx = gamma * (dout * invsig
                  -np.mean(dout * x_centered * invsig**3, axis=0) * x_centered
                  -np.mean(dout * invsig, axis=0)[None,:]
                  +np.mean(np.mean(dout * x_centered * invsig**3, axis=0) * x_centered, axis=0)[None,:])
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    cache = {}
        
    # get sample mean and variance
    sample_mean = np.mean(x, axis=1, keepdims=True) # Shape : (N,)
    sample_var = np.var(x, axis=1, keepdims=True)   # Shape : (N,)
        
    # normalize and scale/shift with sample mean and variance
    xhat = (x - sample_mean) / np.sqrt(sample_var + eps) # Shape : (N, D)
    out = gamma * xhat + beta
        
    # caching neccessary variables for backward pass
    cache['x'] = x
    cache['xhat'] = xhat
    cache['gamma'] = gamma
    cache['eps'] = eps
    cache['sample_mean'] = sample_mean
    cache['sample_var'] = sample_var     
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # get cached items
    x, xhat, gamma, eps, sample_mean, sample_var = \
    cache['x'], cache['xhat'], cache['gamma'], cache['eps'], \
    cache['sample_mean'], cache['sample_var']
    N, D = x.shape
    
    # get gradient for dgamma and dbeta
    dgamma = np.sum(xhat * dout, axis=0)
    dbeta = np.sum(dout, axis=0)
    
    # get gradient for dx
    dxhat = dout * gamma # Shape : (N, D)
    dvar = np.sum(dxhat * (x - sample_mean), axis=1,keepdims=True) * -1/2 * ((sample_var + eps)**(-3/2)) # Shape : (N,)
    dmu = np.sum(dxhat * -1/np.sqrt(sample_var + eps), axis=1, keepdims=True)+ dvar * -2 * np.mean(x - sample_mean, axis=1, keepdims=True) # Shape : (N,)
    dx = dxhat * 1/np.sqrt(sample_var + eps) + dvar * 2 * (x - sample_mean)/D + (dmu/D) # Shape : (N, D)
                  
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        
        out = x
        
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        
        dx = dout * mask
        
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 


    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    
    # get neccessary dimension info
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    # get output dimension 
    H_ = int(1 + (H + 2 * pad - HH) / stride)
    W_ = int(1 + (W + 2 * pad - WW) / stride)
    
    out = np.zeros((N, F, H_, W_))
    
    for i in range(N):
        for j in range(F):
            for k in range(H_):
                for l in range(W_):
                    point_h = stride * k
                    point_w = stride * l
                    for p in range(C):
                        # padding on each color pannel
                        x_pad = np.pad(x[i,p,:,:], ((pad,pad),(pad,pad)),'constant') 
                        # convolution
                        out[i,j,k,l] += np.sum(x_pad[point_h:point_h+HH,point_w:point_w+WW] * w[j,p,:,:])
                    # adding bias 
                    out[i,j,k,l] += b[j]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    
    # get cache values
    x, w, b, conv_param = cache
    pad = conv_param['pad']
    stride = conv_param['stride']
    
    # get shape
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    N, F, H_, W_ = dout.shape
    
    dx_pad = np.zeros((N,C,H+2*pad,W+2*pad)) # Shape : (N, C, H+2*pad, W+2*pad)
    dw = np.zeros_like(w) # Shape : (F, C, HH, WW)
    
    # get dw_revised
    for j in range(F):
        for p in range(C):
            for k in range(H_):
                for l in range(W_):
                    point_h = stride * k
                    point_w = stride * l
                    
                    for i in range(N):
                        x_pad = np.pad(x[i,p,:,:], ((pad,pad),(pad,pad)), 'constant')
                        dw[j,p,:,:] += x_pad[point_h:point_h+HH,point_w:point_w+WW] * dout[i,j,k,l]

    # get dx        
    for i in range(N):
        for p in range(C):
            for j in range(F):
                for k in range(H_):
                    for l in range(W_):
                        point_h = stride * k
                        point_w = stride * l
                        
                        ## get padding mask
                        x_norm = x[i,p,:,:]//x[i,p,:,:]
                        pad_mask = np.pad(x_norm, ((pad,pad),(pad,pad)), 'constant')
                        
                        ## get dx_pad
                        dx_pad[i,p,point_h:point_h+HH,point_w:point_w+WW] += \
                        pad_mask[point_h:point_h+HH,point_w:point_w+WW] * w[j,p,:,:] * dout[i,j,k,l]
    
    ## strip padding from the dx_pad)
    dx = dx_pad[:,:,pad:-pad,pad:-pad] 
    
    # get db
    db = np.sum(dout, axis=(0,2,3)) # Shape : (F,)
                    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    
    # get dimension
    N, C, H, W = x.shape
    
    # get params
    h = pool_param['pool_height']
    w = pool_param['pool_width']
    stride = pool_param['stride']
    
    # initialize pooling output
    H_ = 1 + (H - h) // stride
    W_ = 1 + (W - w) // stride
    out = np.zeros((N,C,H_,W_))
    max_mask = np.zeros_like(x)
    
    # pooling
    for i in range(N):
        for p in range(C):
            for k in range(H_):
                for l in range(W_):
                    h_ = k * stride
                    w_ = l * stride
                    original = x[i,p,h_:h_+h,w_:w_+w] # to cache position of max values
                    out[i,p,k,l] = np.max(original)
                    max_mask[i,p,h_:h_+h,w_:w_+w] = (original == original.max()) * 1 # max value mask
    
    pool_param['max_mask'] = max_mask
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives (Shape : (N,C,H_,W_))
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    
    # get dimension
    N, C, H_, W_ = dout.shape
    
    # get pool_param
    x, pool_param = cache
    h, w, stride, max_mask = pool_param['pool_height'], pool_param['pool_width'], \
    pool_param['stride'], pool_param['max_mask']
    
    # initialize with zero
    dx = np.zeros_like(x)
    
    for i in range(N):
        for p in range(C):
            for k in range(H_):
                for l in range(W_):
                    h_ = k * stride
                    w_ = l * stride
                    dx[i,p,h_:h_+h,w_:w_+w] = max_mask[i,p,h_:h_+h,w_:w_+w] * dout[i,p,k,l]
                    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    
    # get dimension
    N, C, H, W = x.shape
    
    # get batch normalized output
    x_trans_re = x.transpose(1,0,2,3).reshape(-1,C)
    temp, cache = batchnorm_forward(x_trans_re, gamma, beta, bn_param)
    out = temp.reshape(C,N,H,W).transpose(1,0,2,3)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    
    # get dimension
    N, C, H, W = dout.shape
    
    # get batch normalized output
    dout_trans_re = np.transpose(dout, (1,0,2,3)).reshape(-1,C)
    temp_dx, dgamma, dbeta = batchnorm_backward(dout_trans_re, cache)
    dx = temp_dx.reshape(C,N,H,W).transpose((1,0,2,3))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer number of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    
    # get dimension
    N, C, H, W = x.shape
    E = C//G
    
    cache = {}
        
    # get sample mean and variance
    x_ = x.reshape(N*G,-1)
    sample_mean = np.mean(x_, axis=1, keepdims=True) # Shape : (N*G,)
    sample_var = np.var(x_, axis=1, keepdims=True)   # Shape : (N*G,)
    
    # normalize and scale/shift with sample mean and variance
    xhat = (x_ - sample_mean) / np.sqrt(sample_var + eps) # Shape : (N*G, E * H * W)
    xhat = xhat.reshape(N,C,H,W)
    out = gamma * xhat + beta

    # caching neccessary variables for backward pass
    cache['x'] = x
    cache['xhat'] = xhat
    cache['gamma'] = gamma
    cache['eps'] = eps
    cache['sample_mean'] = sample_mean
    cache['sample_var'] = sample_var
    cache['G'] = G
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    
    # get cached items
    x, xhat, gamma, eps, sample_mean, sample_var, G = \
    cache['x'], cache['xhat'], cache['gamma'], cache['eps'], \
    cache['sample_mean'], cache['sample_var'], cache['G']
    N, C, H, W = x.shape
    D = C//G * H * W
    x = x.reshape(N*G,-1)
    
    # get gradient for dgamma and dbeta
    dgamma = np.sum(xhat * dout, axis=(0,2,3), keepdims=True)
    dbeta = np.sum(dout, axis=(0,2,3), keepdims=True)
    
    # get gradient for dx
    dxhat = (dout * gamma).reshape(N*G,-1) # Shape : (N*G,D)
    dvar = np.sum(dxhat * (x - sample_mean), axis=1,keepdims=True) * -1/2 * ((sample_var + eps)**(-3/2)) # Shape : (N*G,)
    dmu = np.sum(dxhat * -1/np.sqrt(sample_var + eps), axis=1, keepdims=True)+ dvar * -2 * np.mean(x - sample_mean, axis=1, keepdims=True) # Shape : (N*G,)
    dx = dxhat * 1/np.sqrt(sample_var + eps) + dvar * 2 * (x - sample_mean)/D + (dmu/D) # Shape : (N*G, D)
    dx = dx.reshape(N,C,H,W) # Shape : (N,C,H,W)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
