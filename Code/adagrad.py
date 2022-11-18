import numpy as np
def adagrad_update(cache,params,grads,learning_rate=0.01):
    for c,param,grad, in zip(cache,params,reversed(grads)):
        for i in range(len(grad)):
            cache[i] += grad[i]**2
            param[i] += - learning_rate * grad[i] / (np.sqrt(cache[i])+1e-8) # for preventing divide by 0