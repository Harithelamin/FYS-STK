import numpy as np
import matplotlib.pyplot as plt



def sgd(x,y,lr,it):
    x = np.array(x)
    ones = np.ones(shape=(x.shape[0],1))
    x = np.concatenate((ones, x), axis=1)
    y = np.array(y)
    W = np.zeros(shape=(x.shape[1],1))
    all_W = list()
    for i in range(it):
        y_p = x @ W
        dW = np.dot(x.T, y_p - y)
        W = W - lr * dW
        all_W.append(W)
    return all_W

x = [[0.56351121, 0.99432814], [0.98461591, 0.2181135], [0.72111435, 0.9264389]]
y = [[0.74592086], [0.3833196], [0.37965535]]
lr = 0.001 # learning_rate
it = 2 # iteration
obj=sgd(x,y,lr,it)
#print(sgd(x,y,lr,it))

plt.plot(obj, label='sgd')
plt.legend()
plt.show()