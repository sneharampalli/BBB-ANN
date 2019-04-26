import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split

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
    m, d = X_train.shape
    reg_lambda = 0.01
    # Stochastic Gradient Descent
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
            der_a_2 = deriv_relu_inter(z_2)
            der_a_2[der_a_2 < 0] = 0
        z_3 = np.dot(a_2, W_2) + b_2
        a_3 = sigmoid(z_3)

        # Back propagation
        delta_3 = a_3 - y_train
        delta_2 = np.multiply(np.dot(delta_3, W_2.T), der_a_2)
        deriv_W2 = np.dot(a_2.T, delta_3)
        deriv_W1 = np.dot(a_1.T, delta_2) 

        deriv_W2 += reg_lambda * W_2
        deriv_W1 += reg_lambda * W_1

        deriv_b1 = np.sum(delta_2, axis=0) 
        deriv_b2 = np.sum(delta_3, axis = 0) 

        W_1 = W_1 - np.multiply(step_size, deriv_W1)
        W_2 = W_2 - np.multiply(step_size, deriv_W2)

        b_1 = b_1 - np.multiply(step_size, deriv_b1)
        b_2 = b_2 - np.multiply(step_size, deriv_b2)
    return W_1, W_2, b_1, b_2

def classification_error(y_pred, y_true):
    err = 1 - (np.sum(y_pred == y_true) / len(y_true))
    return err

if __name__ == "__main__":
    # Importing and cleaning data
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
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    # Parameters we can play around with
    
    num_input_neurons = 30 # number of features
    num_hidden_neurons = 32 # number of neurons in hidden layer
    num_output_neurons = 1 # number of neurons in output layer
    step_size = 0.001
    iterations = 2000 
    func_type = 'sigmoid' # can switch out with "relu" to see the results

    # Split the dataset into 4/5 training, 1/5 testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    m, d = X_train.shape

    # Initialize weight vectors W_1, W_2, b_1, b_2 for a single layer perceptron
    W_1 = np.random.randn(num_input_neurons, num_hidden_neurons) / np.sqrt(num_input_neurons) # weight vector for layer 1
    W_2 = np.random.randn(num_hidden_neurons, num_output_neurons) / np.sqrt(num_hidden_neurons) # weight vector for layer 2
    b_1 = np.zeros((1, num_hidden_neurons)) # bias vector for layer 1
    b_2 = np.zeros((1, num_output_neurons)) # bias vector for layer 2

    [W_1, W_2, b_1, b_2] = single_layer_perceptron(X_train, y_train, step_size, iterations, W_1, W_2, b_1, b_2, func_type)
    print(W_1.shape)
    print(W_2.shape)
    print(b_1.shape)
    print(b_2.shape)
    if (func_type == 'sigmoid'):
        predicted_labels = np.sign(sigmoid(sigmoid(X_train * W_1 + b_1) * W_2 + b_2) - (1/2))
        predicted_test_labels = np.sign(sigmoid(sigmoid(X_test * W_1 + b_1) * W_2 + b_2) - (1/2))
    #     predicted_test_labels = np.sign(sigmoid(W_2 * sigmoid(W_1 * X_test.T + b_1) + b_2) - (1/2)).T;
    else:
        predicted_labels = np.sign(sigmoid(relu(X_train * W_1 + b_1) * W_2 + b_2) - (1/2))
        predicted_test_labels = np.sign(sigmoid(relu(X_test * W_1 + b_1) * W_2 + b_2) - (1/2))
    # #     predicted_test_labels = np.sign(sigmoid(W_2 * relu(W_1 * X_test.T + b_1) + b_2) - (1/2)).T;
    
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == -1:
            predicted_labels[i] = 0
    for i in range(len(predicted_test_labels)):
        if predicted_test_labels[i] == -1:
            predicted_test_labels[i] = 0
    print(predicted_labels)
    train_error = classification_error(predicted_labels, y_train);
    print("TRAIN ERROR: %f" %(train_error))
    test_error = classification_error(predicted_test_labels, y_test);
    print("TEST ERROR: %f" %(test_error))
