import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Step 2: Add a bias term (column of ones) to the input data
X_b = np.c_[np.ones((100, 1)), X]

# Step 3: Define the learning rate and number of iterations
learning_rate = 0.1
n_iterations = 1000
m = len(X_b)

# Step 4: Initialize random weights
theta = np.random.randn(2, 1)

# Step 5: Gradient Descent Algorithm
cost_history = []

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - learning_rate * gradients
    cost = np.mean((X_b.dot(theta) - y) ** 2)
    cost_history.append(cost)

# Step 6: Plotting the cost function over iterations
plt.plot(range(n_iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.show()

# Step 7: Plotting the fitted line
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, X_b.dot(theta), color='red', label='Fitted Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Gradient Descent')
plt.legend()
plt.show()

print(f"Fitted Parameters (Theta): {theta}")