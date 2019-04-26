import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split

d1 = 10

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def deriv_sigmoid(x):
    return sigmoid(x) - np.square(sigmoid(x))

# Relu function
def relu(x):
    return np.max(0, x)        

# Derivative of relu function
def deriv_relu(x):
    return np.sign(x)

def single_layer_perceptron(X_train, y_train, step_size, iterations, W_1, W_2, b_1, b_2, func_type):
    # Stochastic Gradient Descent
    for x in range(iterations):
        z_1 = W_1 * X_train.T + b_1
        if func_type is 'sigmoid':
            a_1 = sigmoid(z_1)
            g_der = deriv_sigmoid(z_1)
        else: 
            a_1 = relu(z_1)
            g_der = deriv_relu_inter(z_1)
            g_der[g_der < 0] = 0
        f_wb = W_2 * a_1 + b_2
        g = sigmoid(f_wb)  

        deriv_W1 = np.multiply((W_2.T * (g - y_train.T)), g_der) * X_train / m;
        deriv_W2 = (g - y_train.T) * a_1.T / m

        deriv_b1 = np.multiply((g - y_train.T) * g_der.T, W_2) / m
        deriv_b2 = sum(g - y_train.T) / m

        W_1 = W_1 - np.multiply(step_size, deriv_W1)
        W_2 = W_2 - np.multiply(step_size, deriv_W2)

        b_1 = b_1 - np.multiply(step_size, deriv_b1).T
        b_2 = b_2 - np.multiply(step_size, deriv_b2)
    return W_1, W_2, b_1, b_2

def classification_error(y_pred, y_true):
    err = 1 - (np.sum(y_pred == y_true) / len(y_true))
    return err

# Data cleaning
num_remove = 1
data = []
labels = []
with open('wdbc.dat', "r") as data_file:
    for line in data_file:
        # get last char
        line = line.strip()
        # Converting the labels to 0 and 1 for binary classification
        binary_label = 0 if line[-(num_remove):] is 'M' else 1
        labels.append(binary_label)
        line = line[:-(num_remove + 1)]
        features = [float(x) for x in line.split(",")]
        data.append(features)
    c = list(zip(data, labels))
    X, y = zip(*c)
    y = np.matrix(y).T
    X = np.matrix(X)

# Normalization of the data --> raises the error?
# mean = X.mean(axis=0)
# std = X.std(axis=0)
# X = (X - mean) / std

# Parameters we can play around with
step_size = 0.1
iterations = 1000 
func_type = "sigmoid" # can switch out with "relu" to see the results

# Split the dataset into 2/3 training, 1/3 testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

m, d = X_train.shape

# Initialize weight vectors W_1, W_2, b_1, b_2 for a single layer perceptron
W_1 = np.random.rand(d1, d)
W_2 = np.random.rand(1, d1)
b_1 = np.zeros((d1, 1))
b_2 = np.zeros(1)

[W_1, W_2, b_1, b_2] = single_layer_perceptron(X_train, y_train, step_size, iterations, W_1, W_2, b_1, b_2, func_type)
if (func_type == 'sigmoid'):
    predicted_labels = np.sign(sigmoid(W_2 * sigmoid(W_1 * X_train.T + b_1) + b_2) - (1/2)).T;
#     predicted_test_labels = np.sign(sigmoid(W_2 * sigmoid(W_1 * X_test.T + b_1) + b_2) - (1/2)).T;
else:
    predicted_labels = np.sign(sigmoid(W_2 * relu(W_1 * X_train.T + b_1) + b_2) - (1/2)).T;
#     predicted_test_labels = np.sign(sigmoid(W_2 * relu(W_1 * X_test.T + b_1) + b_2) - (1/2)).T;
    
train_error = classification_error(predicted_labels, y_train);
print("TRAIN ERROR: %f" %(train_error))
