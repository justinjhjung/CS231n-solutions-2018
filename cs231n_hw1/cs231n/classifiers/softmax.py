import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    num_examples = X.shape[0]
    num_dim = W.shape[1]
    
    for i in range(num_examples):
        # get score 
        score = X[i].dot(W) # shape = (1, C)
        score -= np.max(score) # regularized score
        exp_score = np.exp(score)
        
        # get loss 
        exp_sum = np.sum(exp_score) 
        exp_yi = exp_score[y[i]]
        L_i = -np.log(exp_yi/exp_sum)
        loss += L_i/num_examples 
        
        # get gradient
        for j in range(num_dim):
            if j == y[i]:
                dW[:,j] += X[i] * (-1 + exp_score[j]/exp_sum) / num_examples
            else:
                dW[:,j] += X[i] * (exp_score[j]/exp_sum) / num_examples 

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    
    num_examples = X.shape[0]
    
    # get score
    score = X.dot(W)
    score -= np.amax(score, axis=1)[:,None]
    exp_score = np.exp(score)
    
    # get loss
    exp_sum = np.sum(exp_score, axis=1)
    exp_y = exp_score[np.arange(num_examples), y]
    loss = -np.sum(np.log(exp_y/exp_sum)) / num_examples

    # get gradient
    temp = exp_score/exp_sum[:,None]
    temp[np.arange(num_examples),y] -= 1
    dW = X.T.dot(temp)
    dW /= num_examples
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

