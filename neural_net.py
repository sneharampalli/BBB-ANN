import numpy as np
import pandas as pd
import json

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def deriv_sigmoid(x):
    return sigmoid(x) - np.square(sigmoid(x))

# Relu function
def relu(x):
    return np.maximum(0, x)        

# Derivative of relu function
def deriv_relu(x):
    return np.sign(x)

def multi_layer_perceptron(X_train, y_train, step_size, iterations, W_1, W_2, b_1, b_2, func_type):
    y_train = (y_train + 1)/2

    m, d = X_train.shape
    reg_lambda = 0.01
    # Batch Gradient Descent
    for x in range(iterations):
        # Forward propagation 
        a_1 = X_train
        z_2 = np.dot(a_1, W_1) + b_1
        g_der = 0
        if func_type is 'sigmoid':
            a_2 = sigmoid(z_2)
            der_a_2 = deriv_sigmoid(z_2)           
        else: 
            a_2 = relu(z_2)
            der_a_2 = deriv_relu(z_2)
            der_a_2[der_a_2 < 0] = 0
        z_3 = np.dot(a_2, W_2) + b_2
        a_3 = sigmoid(z_3)

        # Back propagation
        delta_3 = a_3 - y_train
        delta_2 = np.multiply(np.dot(delta_3, W_2.T), der_a_2)
        deriv_W2 = np.dot(a_2.T, delta_3) / m
        deriv_W1 = np.dot(a_1.T, delta_2) / m

        deriv_W2 += reg_lambda * W_2
        deriv_W1 += reg_lambda * W_1

        deriv_b1 = np.sum(delta_2, axis=0) / m
        deriv_b2 = np.sum(delta_3, axis=0) / m

        W_1 = W_1 - np.multiply(step_size, deriv_W1)
        W_2 = W_2 - np.multiply(step_size, deriv_W2)

        b_1 = b_1 - np.multiply(step_size, deriv_b1)
        b_2 = b_2 - np.multiply(step_size, deriv_b2)
    return W_1, W_2, b_1, b_2

def classification_error(y_pred, y_true):
    err = 1 - (np.sum(y_pred == y_true) / len(y_true))
    return err