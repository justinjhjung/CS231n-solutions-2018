import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)  # shape = (1, C)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                # dloss/dmargin * dmargin/dscores[j] * dscores[j]/dW = 1 * 1 * X[i] = X[i]
                dW[:,j] += X[i]    
                # dloss/dmargin * dmargin/dcorrect_class_score * dcorrect_class_score/dW = 1 * -1 * X[i] = -X[i] 
                dW[:,y[i]] -= X[i] 

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################


    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    
    num_train = X.shape[0]
    
    # Getting score
    scores = X.dot(W)
    
    # Getting margins
    correct_class_scores = scores[np.arange(num_train), y].reshape(-1,1)
    margins = np.maximum(0, scores - correct_class_scores + 1) # note delta = 1
    mask_cor = (margins == 1)
    margins[np.arange(num_train), y] = 0 # let margins of correct classes be 0
    
    # Getting loss
    loss = np.sum(margins)/num_train 
    loss += reg * np.sum(W * W)
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    
    # Get dW
    mask_pos = (margins > 0) * 1
    correct_indicators = np.sum(mask_pos, axis=1) # This is to calculate dloss/dW_yi: dloss/dW_yi = - sum(1(>0)) * X[i] 
    mask_pos[mask_cor] -= correct_indicators.T # Since only elements influenced by dloss/dW_yi have negative sign, we do it upfront.
    dW += np.dot(X.T, mask_pos) # - sum(1(>0)) is already reflected, now the only thing we should do is to calculate X.T.dot(mask_pos)
    dW /= num_train
    
    # Add derivative of regularization term with respect to W
    
    dW += reg * 2 * W
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW

