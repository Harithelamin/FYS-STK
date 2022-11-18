def rmsprop_update(cache,params,grads,learning_rate=0.01,decay_rate=0.9):
    for c,param,grad, in zip(cache,params,reversed(grads)):
        for i in range(len(grad)):
            cache[i] = decay_rate * cache[i] + (1-decay_rate) * grad[i]**2
            param[i] += - learning_rate * grad[i] / (np.sqrt(cache[i])+1e-4)