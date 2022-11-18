import numpy as np
import matplotlib.pyplot as plt


def gradient_descent(gradient, start, learn_rate, n_iter=50, tolerance=1e-06):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(x,y, 'r')
    vector = start
    x1=[vector]
    y1=[vector**2]
    for _ in range(n_iter):
        
        diff = -learn_rate * gradient(vector)
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff
        
        x1.append(vector)
        y1.append(vector**2)
    
    for i in range(len(x1)):
         plt.plot(x1[i], y1[i], marker='o', markersize=3, color="blue")
    plt.plot(x1,y1)
    plt.show()
    return vector

if __name__=="__main__":
    gradient=lambda v: 2 * v
    learn_rate=0.0001
    start=10.0


    x = np.array([47.72939999, 23.42680403, 61.53035803, 19.47563963, 64.81320787])
    y = np.array([71.70700585, 38.77759598, 52.5623823 , 48.54663223, 33.23092513])
    gradient_descent( gradient , start=start, learn_rate= learn_rate)