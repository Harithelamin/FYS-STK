import numpy as np
import matplotlib.pyplot as plt
import StatFunctions

# Data
x = np.array([47.72939999, 23.42680403, 61.53035803, 19.47563963, 64.81320787])
y = np.array([71.70700585, 38.77759598, 52.5623823 , 48.54663223, 33.23092513])


# Gradient Descent Function
def plant_gd(x, y, iterations = 1000, learning_rate = 0.0001,
					stopping_threshold = 1e-6):
	
	# Hyperparameters
	current_weight = 0.1
	current_bias = 0.01
	iterations = iterations
	learning_rate = learning_rate
	n = float(len(x))
	
	costs = []
	weights = []
	previous_cost = None
	
	# Estimation of optimal parameters
	for i in range(iterations):
		
		# Making predictions
		y_predicted = (current_weight * x) + current_bias
		
		# Calculationg the current cost
		current_cost = StatFunctions.mean_squared_error(y, y_predicted)

		# If the change in cost is less than or equal to
		# stopping_threshold we stop the gradient descent
		if previous_cost and abs(previous_cost-current_cost)<=stopping_threshold:
			break
		
		previous_cost = current_cost

		costs.append(current_cost)
		weights.append(current_weight)
		
		# Calculating the gradients
		weight_derivative = -(2/n) * sum(x * (y-y_predicted))
		bias_derivative = -(2/n) * sum(y-y_predicted)
		
		# Updating weights and bias
		current_weight = current_weight - (learning_rate * weight_derivative)
		current_bias = current_bias - (learning_rate * bias_derivative)
				
		# Printing the parameters for each 1000th iteration
		print(f"Iteration {i+1}: Cost {current_cost}, Weight \
		{current_weight}, Bias {current_bias}")
	
	

    # Visualizing the weights and cost at for all iterations
	plt.figure(figsize = (8,6))
	plt.plot(weights, costs)
	plt.scatter(weights, costs, marker='o', color='red')
	plt.title("Cost vs Weights")
	plt.ylabel("Cost")
	plt.xlabel("Weight")
	plt.show()
	return current_weight, current_bias

	
if __name__=="__main__":
   
    # weight and bias using gradient descent
	estimated_weight, eatimated_bias = plant_gd(x, y, iterations=2000)

	print(f"Estimated Weight: {estimated_weight}\nEstimated Bias: {eatimated_bias}")

	# Making predictions using estimated parameters
	Y_pred = estimated_weight*x + eatimated_bias

	# Plotting the regression line
	plt.figure(figsize = (8,6))
	plt.scatter(x, y, marker='o', color='red')
	plt.plot([min(x), max(y)], [min(Y_pred), max(Y_pred)], color='blue',markerfacecolor='red',
			markersize=10,linestyle='dashed')
	plt.xlabel("x")
	plt.ylabel("Y")
	plt.show()
