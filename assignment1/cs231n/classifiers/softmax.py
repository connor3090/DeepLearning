from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW = np.zeros(W.shape) # initialize the gradient as zero
    h = 0.00001
    # compute the loss and the gradient
    num_classes = W.shape[1] # C, 1 is columns
    num_train = X.shape[0] # N, 0 is rows
    loss = 0.0
    for i in range(num_train):
      # Normalize the scores to prevent numeric instability
      scores = X[i].dot(W)        
      scores -= np.max(scores)        
      p = np.exp(scores) / np.sum(np.exp(scores))      
      correct_class_score = p[y[i]]
      loss += -np.log(correct_class_score)
      for j in range(num_classes):
        dW[:,j] += X[i] * p[j]
      dW[:,y[i]] -= X[i]
    loss /= num_train
    loss += reg * np.sum(W*W)

    dW /= num_train
    dW += reg * 2 * W
            

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    scores = X.dot(W)
    num_rows = scores.shape[0]
    scores -= np.max(scores, axis=1).reshape(num_rows,1)
    p = np.exp(scores) / np.sum(np.exp(scores), axis = 1).reshape(num_rows,1)
    correct_p = p[(np.arange(num_rows)),(y)]
    log_p = np.log(correct_p)
    loss = -np.sum(log_p) / num_rows
    loss += reg * np.sum(W*W)

    p[np.arange(num_rows),y] -= 1
    dW = X.transpose().dot(p)
    dW /= num_rows
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss, dW
