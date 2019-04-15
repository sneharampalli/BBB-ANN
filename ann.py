# All import statements
import numpy as np
import pandas as pd 
import json

# Number of iterations set to 1000
iterations = 1000
d1 = 10
func_type = 'sigmoid'

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Derivative of sigmoid function
def deriv_sigmoid(x):
    return sigmoid(x) - sigmoid(x)**2

# Relu function
def relu(x):
    return np.max(0, x)        

# Derivative of relu function
def deriv_relu(x):
    return np.sign(x)

# Cleaning up the data
num_remove = 1
data = []
labels = []
with open('wbdc.dat', "r") as data_file:
    for line in data_file:
        # get last char
        line = line.strip()
        binary_label = 1 if line[-(num_remove):] is 'M' else 0
        labels.append(binary_label)
        line = line[:-(num_remove + 1)]
        features = [float(x) for x in line.split(",")]
        data.append(features)
    c = list(zip(data, labels))
    X, y = zip(*c)
    y = np.matrix(y).T
    X = np.matrix(X)

mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

# Initialize weight vectors W_1, W_2, b_1, b_2
W_1 = np.random.randn(0, 4)
W_2 = np.random.randn(0, 4)
b_1 = np.zeros((d1, 1))
b_2 = np.zeros(1)

# Stochastic Gradient Descent
for x in range(iterations):
    z_1 = W_1 * X.T + b_1
    if (func_type is 'sigmoid')
        a_1 = sigmoid(z_1)
        g_der = deriv_sigmoid(z_1)
    else: 
        a_1 = relu(z_1)
        g_der = deriv_relu_inter(z_1)
        g_der[g_der < 0] = 0

    f_wb = W_2 * a_1 + b_2
    g = sigmoid(f_wb)  
    
    deriv_W1 = (W_2.T * (g - y.T)) .* g_der * X / m
    deriv_W2 = (g - y.T) * a_1.T / m
    
    deriv_b1 = (g - y.T) * g_der.T .* W_2 / m
    deriv_b2 = sum(g - y.T) / m

    W_1 = W_1 - (step_size .* deriv_W1)
    W_2 = W_2 - (step_size .* deriv_W2)

    b_1 = b_1 - (step_size * deriv_b1).T
    b_2 = b_2 - (step_size .* deriv_b2)