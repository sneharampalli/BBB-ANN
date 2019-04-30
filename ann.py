import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import neural_net as nn

# This function runs leave one out (569 fold) cross validation on our nueral network
def leave_one_out_cross_validation(X, y):
    for i in range(len(X)):
        W_1 = np.random.randn(num_input_neurons, num_hidden_neurons) / np.sqrt(num_input_neurons) # weight vector for layer 1
        W_2 = np.random.randn(num_hidden_neurons, num_output_neurons) / np.sqrt(num_hidden_neurons) # weight vector for layer 2
        b_1 = np.zeros((1, num_hidden_neurons)) # bias vector for layer 1
        b_2 = np.zeros((1, num_output_neurons)) # bias vector for layer 2
        leave_out_index = i # Data entry index that we leave out for this iteration of CV
        X_test = X[i] 
        y_test = y[i]
        sub_X_train_1 = X[:i]
        sub_X_train_2 = X[i+1:]
        X_train = np.concatenate((sub_X_train_1, sub_X_train_2)) # Build training set out of all but data entry i
        sub_y_train_1 = y[:i]
        sub_y_train_2 = y[i+1:]
        y_train = np.concatenate((sub_y_train_1, sub_y_train_2)) # Build training set out of all but data entry i
        [W_1, W_2, b_1, b_2] = nn.multi_layer_perceptron(X_train, y_train, 0.0003, 2000, W_1, W_2, b_1, b_2, 'sigmoid')
    
        predicted_labels = np.sign(nn.sigmoid(nn.sigmoid(X_train * W_1 + b_1) * W_2 + b_2) - (1/2))
        predicted_test_labels = np.sign(nn.sigmoid(nn.sigmoid(X_test * W_1 + b_1) * W_2 + b_2) - (1/2))
        
        # Keep track of number of folds that resulted in correct prediction
        if predicted_test_labels[0] == y_test:
            num_correct += 1
    print("OVERALL ACCURACY:: %f" %(num_correct / len(X)))
    
# Creating 20 folds of X data and its labels (both training and testing data)
def build_even_datasets(X, y, folds):
    rows, cols = y.shape
    neg_ones = np.where(y == -1)[0]  # Count of negative ones in X
    pos_ones = np.where(y == 1)[0] # Count of positive ones in X
    np.random.shuffle(neg_ones) # Get random shuffle of data
    np.random.shuffle(pos_ones)
    neg_splits = np.array_split(neg_ones, folds)
    pos_splits = np.array_split(pos_ones, folds)
    X_splits = []
    y_splits = []
    for i in range(folds): # Create new fold with more even split of data
        fold = np.concatenate([pos_splits[i], neg_splits[i]]) 
        np.random.shuffle(fold)
        X_splits.append(np.asarray(X[fold]))
        y_splits.append(np.asarray(y[fold]))
    return X_splits, y_splits

# 20-fold cross validation on both learning rates and hidden units 
def twenty_fold_CV(X_splits, y_splits, lr, hidden_units, folds):
    tr_er = 0
    tes_er = 0
    f_one = 0
    num_input_neurons = 30 # number of features
    num_hidden_neurons = hidden_units # number of neurons in hidden layer
    num_output_neurons = 1 # number of neurons in output layer
    step_size = lr
    iterations = 3000 
    func_type = 'sigmoid' # can switch out with "relu" to see the results
    
    for i in range(folds):
        X_test = X_splits[i]
        y_test = y_splits[i]
        X_train = []
        y_train = []
        for j in range(folds):
            if j != i:
                X_train.extend(X_splits[j]) # Concatenate other folds into train data
                y_train.extend(y_splits[j]) # Concatenate other folds into train labels

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        m, d = X_train.shape

        # Initialize weight vectors W_1, W_2, b_1, b_2 for a multi layer perceptron
        W_1 = np.random.randn(num_input_neurons, num_hidden_neurons) / np.sqrt(num_input_neurons) # weight vector for layer 1
        W_2 = np.random.randn(num_hidden_neurons, num_output_neurons) / np.sqrt(num_hidden_neurons) # weight vector for layer 2
        b_1 = np.zeros((1, num_hidden_neurons)) # bias vector for layer 1
        b_2 = np.zeros((1, num_output_neurons)) # bias vector for layer 2
        [W_1, W_2, b_1, b_2] = nn.multi_layer_perceptron(X_train, y_train, step_size, iterations, W_1, W_2, b_1, b_2, func_type)
        if (func_type == 'sigmoid'):
            predicted_labels = np.sign(nn.sigmoid(np.dot(nn.sigmoid(np.dot(X_train, W_1) + b_1), W_2) + b_2) - (1/2))
            predicted_test_labels = np.sign(nn.sigmoid(np.dot(nn.sigmoid(np.dot(X_test, W_1) + b_1), W_2) + b_2) - (1/2))
        else:
            predicted_labels = np.sign(nn.sigmoid(nn.relu(np.dot(X_train, W_1) + b_1) * W_2 + b_2) - (1/2))
            predicted_test_labels = np.sign(nn.sigmoid(np.dot(nn.relu(np.dot(X_test, W_1)  + b_1), W_2) + b_2) - (1/2))
        
        train_error = nn.classification_error(predicted_labels, y_train)
        tr_er += train_error
        test_error = nn.classification_error(predicted_test_labels, y_test)
        tes_er += test_error
        f1 = f1_score(y_test, predicted_test_labels)
        f_one += f1
    print("OVERALL TRAIN ERROR: %f" %(tr_er / folds))
    print("OVERALL TEST ERROR: %f" %(tes_er / folds))
    print("OVERALL F1 SCORE: %f" %(f_one / folds))
    return (tr_er / folds), (tes_er / folds), (f_one / folds)

if __name__ == "__main__":
    # Importing and cleaning data
    num_remove = 1
    data = []
    labels = []
    with open('wdbc.dat', "r") as data_file:
        for line in data_file:
            # get last char
            line = line.strip()
            # Converting the labels to -1 and 1 for binary classification
            binary_label = -1 if line[-(num_remove):] is 'M' else 1
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

    # # Parameters we can play around with
    # num_input_neurons = 30 # number of features
    # num_hidden_neurons = 16 # number of neurons in hidden layer
    # num_output_neurons = 1 # number of neurons in output layer
    # step_size = 0.03
    # iterations = 3000 
    # func_type = 'sigmoid' # can switch out with "relu" to see the results

    # leave_one_out_cross_validation(X, y) # This calls leave one out CV, comment out all code below before running this!

    folds = 20 # num of folds we want to create    
    train_errs = np.zeros(40)
    test_errs = np.zeros(40)
    f1_scores = np.zeros(40)
    learning_rate = [1, 0.1, 0.01, 0.005, 0.001]
    hidden_units = [4, 8, 12, 16, 21, 26, 32, 48]
    i = 0
    for lr in learning_rate:
        for hid in hidden_units:
            X_splits, y_splits = build_even_datasets(X, y, folds)
            train_errs[i], test_errs[i], f1_scores[i] = twenty_fold_CV(X_splits, y_splits, lr, hid, folds)
            i = i + 1

    # # Initialize weight vectors W_1, W_2, b_1, b_2 for a multi layer perceptron
    # W_1 = np.random.randn(num_input_neurons, num_hidden_neurons) / np.sqrt(num_input_neurons) # weight vector for layer 1
    # W_2 = np.random.randn(num_hidden_neurons, num_output_neurons) / np.sqrt(num_hidden_neurons) # weight vector for layer 2
    # b_1 = np.zeros((1, num_hidden_neurons)) # bias vector for layer 1
    # b_2 = np.zeros((1, num_output_neurons)) # bias vector for layer 2

    # [W_1, W_2, b_1, b_2] = nn.multi_layer_perceptron(X_train, y_train, step_size, iterations, W_1, W_2, b_1, b_2, func_type)
    # if (func_type == 'sigmoid'):
    #     predicted_labels = np.sign(nn.sigmoid(nn.sigmoid(X_train * W_1 + b_1) * W_2 + b_2) - (1/2))
    #     predicted_test_labels = np.sign(nn.sigmoid(nn.sigmoid(X_test * W_1 + b_1) * W_2 + b_2) - (1/2))
    # else:
    #     predicted_labels = np.sign(nn.sigmoid(nn.relu(X_train * W_1 + b_1) * W_2 + b_2) - (1/2))
    #     predicted_test_labels = np.sign(nn.sigmoid(nn.relu(X_test * W_1 + b_1) * W_2 + b_2) - (1/2))
    
    # train_error = nn.classification_error(predicted_labels, y_train)
    # print("TRAIN ERROR: %f" %(train_error))
    # test_error = nn.classification_error(predicted_test_labels, y_test)
    # print("TEST ERROR: %f" %(test_error))
    # print("F1 score: %f" %(f1_score(y_test, predicted_test_labels)))