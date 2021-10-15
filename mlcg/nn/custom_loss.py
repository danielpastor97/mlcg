import numpy as np
import torch

class my_loss():
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

    def __init__(self, data, weights=None,**kwargs):
        super(my_loss,self).__init__(weights)
        self.loss_functions = []
        for key in kwargs.keys():
            self.loss_functions.append(key)
        self.n_loss_functions = len(self.loss_functions)
        self.weights = weights
        # 
        if self.weights is not None:
            if np.sum(self.weights) != 1:
                raise ValueError('weights of loss must sum to 1')
        else:
            self.weights = np.ones(len(self.n_loss_functions))/self.n_loss_functions

    def MSELoss(self):
        total_loss = 0
        for ilf,loss_function in enumerate(self.loss_functions):
            params = self.loss_function.values()
            # loss = 
            total_loss += self.weights[ilf]*loss
        return total_loss