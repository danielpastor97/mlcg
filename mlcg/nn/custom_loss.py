import numpy as np
import torch

def my_loss(data,weights=None,**kwargs):
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
    loss_functions = []
    for key,value in kwargs.items():
        loss_functions.append(key)
    n_loss_functions = len(loss_functions)

    # 
    if weights is not None:
        if np.sum(weights) != 1:
            raise ValueError('weights of loss must sum to 1')
    else:
        weights = np.ones(len(n_loss_functions))/n_loss_functions

    loss = 0
    for ilf,loss_function in enumerate(loss_functions):
        loss += weights[ilf]
    return loss