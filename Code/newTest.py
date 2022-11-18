
import numpy as np
from matplotlib import cm
import numpy as np
import StatFunctions
import matplotlib.pyplot as plt

lr = 0.001 # learning_rate
it = 8 # iteration
xx = [[0.56351121, 0.99432814], [0.98461591, 0.2181135], [0.72111435, 0.9264389]]
yy = [[0.74592086], [0.3833196], [0.37965535]]



N = 100
polynomialDegrees = 5
x = np.random.uniform(0, 1, N)
y = np.random.uniform(0, 1, N)
z = StatFunctions.FrankeFunctionWithNoise(x,y,0.0) #adding some noise to the data
polydegree_bootstrap, error_bootstrap, bias_bootstrap, variance_bootstrap = StatFunctions.PolynomialOLSBootstrapResampling(x,y,z,0.2,polynomialDegrees,N)
polydegree_crossValidation, error_crossValidation = StatFunctions.PolynomialOLSCrossValidation(x,y,z,polynomialDegrees,20)
GSD= StatFunctions.sgd(xx, yy, lr, it)
plt.plot(polydegree_bootstrap, error_crossValidation, label='Crossvalidation')
plt.plot(polydegree_bootstrap, error_bootstrap, label='Bootstrap')
plt.plot(GSD)
plt.legend()
plt.show()

