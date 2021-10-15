import numpy as np
import torch

class my_loss(data,weights=None,**kwargs):
    '''
        Custom loss function to 

        Parameters
        ------
        data : {dict}
        weights : {array}
            linear additive term of loss

        Returns
        -------
        loss : {float} 
            error of prediction
    '''

    def __init__(self, weights=None):
        super(my_loss,self).__init__(weights)
        self.loss_functions = []
        for key,value in kwargs.items():
            self.loss_functions.append(key)
        self.n_loss_functions = len(loss_functions)
        self.weights = weights

    # 
    if weights is not None:
        if np.sum(weights) != 1:
            raise ValueError('weights of loss must sum to 1')
    else:
        weights = np.ones(len(n_loss_functions))/n_loss_functions

    total_loss = 0
    for ilf,loss_function in enumerate(loss_functions):
        loss =             
        total_loss += weights[ilf]*loss
    return total_loss