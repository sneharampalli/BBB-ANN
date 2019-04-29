import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import neural_net as nn

def balance_datasets(X, y):
    num_neg_ones = (y == -1).sum() # num of -1 labels = 212
    num_pos_ones = (y == 1).sum() # num of +1 labels = 357
    data_df = pd.concat([pd.DataFrame(np.array(X)), pd.DataFrame(np.array(y))], axis = 1)
    data_df.columns = ['X', 'y']
    # data_df.sort_values(by=['y'])
    print(data_df)
    # len_X = X.shape[0] # num of training examples
    
def leave_one_out_cross_validation(X, y):
    for i in range(len(X)):
        W_1 = np.random.randn(num_input_neurons, num_hidden_neurons) / np.sqrt(num_input_neurons) # weight vector for layer 1
        W_2 = np.random.randn(num_hidden_neurons, num_output_neurons) / np.sqrt(num_hidden_neurons) # weight vector for layer 2
        b_1 = np.zeros((1, num_hidden_neurons)) # bias vector for layer 1
        b_2 = np.zeros((1, num_output_neurons)) # bias vector for layer 2
        leave_out_index = i
        X_test = X[i]
        y_test = y[i]
        sub_X_train_1 = X[:i]
        sub_X_train_2 = X[i+1:]
        X_train = np.concatenate((sub_X_train_1, sub_X_train_2))
        sub_y_train_1 = y[:i]
        sub_y_train_2 = y[i+1:]
        y_train = np.concatenate((sub_y_train_1, sub_y_train_2))
        [W_1, W_2, b_1, b_2] = nn.multi_layer_perceptron(X_train, y_train, 0.0003, 2000, W_1, W_2, b_1, b_2, 'sigmoid')
    
        predicted_labels = np.sign(nn.sigmoid(nn.sigmoid(X_train * W_1 + b_1) * W_2 + b_2) - (1/2))
        predicted_test_labels = np.sign(nn.sigmoid(nn.sigmoid(X_test * W_1 + b_1) * W_2 + b_2) - (1/2))
        if y_test == 1:
            num1s += 1
        elif y_test == -1:
            num0s += 1
        if predicted_test_labels[0] == y_test:
            num_correct += 1
        train_error = classification_error(predicted_labels, y_train)
        train_error = int(train_error * 10000)
        print("Iteration: " + str(i) + " Train error: " + str(train_error/10000) + " " + str(y_test) + " " + str(predicted_test_labels[0][0]))

# CREATE FOLDS BY SPLITTING EACH MALIGNANT AND BENIGN INTO 20 AND PAIRING

def build_even_datasets(X, y, folds):
    rows, cols = y.shape
    neg_ones = np.where(y == -1)[0]
    pos_ones = np.where(y == 1)[0]
    even_neg_ones = np.array_split(neg_ones, folds)
    even_pos_ones = np.array_split(pos_ones, folds)
    X_index_splits = []
    X_splits = []
    for i in range(folds): # getting indices of even splits
        X_index_splits.append(np.concatenate([even_neg_ones[i], even_pos_ones[i]]))
    for i in range(folds):
        curr_split = []
        for j in range(len(X_index_splits[i])):
            curr_split.extend(X[X_index_splits[i][j]])
        X_splits.append(curr_split)

    y_index_splits = []
    y_splits = []
    for i in range(folds): # getting indices of even splits
        y_index_splits.append(np.concatenate([even_neg_ones[i], even_pos_ones[i]]))
    for i in range(folds):
        curr_split = []
        for j in range(len(y_index_splits[i])):
            curr_split.extend(y[y_index_splits[i][j]])
        y_splits.append(curr_split)
    print(len(X))
    print(len(y))
    X_train = X[:round(len(X) * (4 / 5))]
    X_test = X[round(len(X) * (4 / 5)) + 1:]
    y_train = y[:round(len(y) * (4 / 5))]
    y_test = y[round(len(y) * (4 / 5)) + 1:]
    return X_train, X_test, y_train, y_test

def cross_validation(folds, parameters): # parameters = num of units in hidden layer, learning rate;
    # for j in range(len(parameters)):
    #     for i in range(0, len(X), 20):
    #         X_test = X[i:i+20]
    #         y_test = y[i:i+20]
    #         sub_X_train_1 = X[:i]
    #         sub_X_train_2 = X[i+21:]
    #         X_train = np.concatenate((sub_X_train_1, sub_X_train_2))
    #         sub_y_train_1 = y[:i]
    #         sub_y_train_2 = y[i+21:]
    #         y_train = np.concatenate((sub_y_train_1, sub_y_train_2))

    #         # Initialize weight and bias vectors
    #         W_1 = np.random.randn(num_input_neurons, parameters(j)) / np.sqrt(num_input_neurons) # weight vector for layer 1
    #         W_2 = np.random.randn(parameters(j), num_output_neurons) / np.sqrt(num_hidden_neurons) # weight vector for layer 2
    #         b_1 = np.zeros((1, parameters(j))) # bias vector for layer 1
    #         b_2 = np.zeros((1, num_output_neurons)) # bias vector for layer 2

    #         [W_1, W_2, b_1, b_2] = multi_layer_perceptron(X_train, y_train, 0.0003, 2000, W_1, W_2, b_1, b_2, 'sigmoid')
    #         predicted_labels = np.sign(sigmoid(sigmoid(X_train * W_1 + b_1) * W_2 + b_2) - (1/2))
    #         predicted_test_labels = np.sign(sigmoid(sigmoid(X_test * W_1 + b_1) * W_2 + b_2) - (1/2))
    #         test_error = test_error + classification_error(predicted_test_labels, y_test)
    #     test_error_fold = test_error / len(parameters); 
        print("This fold's test error was: ")

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

    # Parameters we can play around with
    
    num_input_neurons = 30 # number of features
    num_hidden_neurons = 32 # number of neurons in hidden layer
    num_output_neurons = 1 # number of neurons in output layer
    step_size = 0.1
    iterations = 2000 
    func_type = 'sigmoid' # can switch out with "relu" to see the results

    # leave_one_out_cross_validation(X, y) # This calls leave one out CV, comment out all code below before running this!

    # Split the dataset into 4/5 training, 1/5 testing
    folds = 20 # num of folds we want to create    
    X_train, X_test, y_train, y_test = build_even_datasets(X, y, folds)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    m, d = X_train.shape

    # Initialize weight vectors W_1, W_2, b_1, b_2 for a multi layer perceptron
    W_1 = np.random.randn(num_input_neurons, num_hidden_neurons) / np.sqrt(num_input_neurons) # weight vector for layer 1
    W_2 = np.random.randn(num_hidden_neurons, num_output_neurons) / np.sqrt(num_hidden_neurons) # weight vector for layer 2
    b_1 = np.zeros((1, num_hidden_neurons)) # bias vector for layer 1
    b_2 = np.zeros((1, num_output_neurons)) # bias vector for layer 2

    [W_1, W_2, b_1, b_2] = nn.multi_layer_perceptron(X_train, y_train, step_size, iterations, W_1, W_2, b_1, b_2, func_type)
    if (func_type == 'sigmoid'):
        predicted_labels = np.sign(nn.sigmoid(nn.sigmoid(X_train * W_1 + b_1) * W_2 + b_2) - (1/2))
        predicted_test_labels = np.sign(nn.sigmoid(nn.sigmoid(X_test * W_1 + b_1) * W_2 + b_2) - (1/2))
    else:
        predicted_labels = np.sign(nn.sigmoid(nn.relu(X_train * W_1 + b_1) * W_2 + b_2) - (1/2))
        predicted_test_labels = np.sign(nn.sigmoid(nn.relu(X_test * W_1 + b_1) * W_2 + b_2) - (1/2))
    
    train_error = nn.classification_error(predicted_labels, y_train)
    print("TRAIN ERROR: %f" %(train_error))
    test_error = nn.classification_error(predicted_test_labels, y_test)
    print("TEST ERROR: %f" %(test_error))
    print("F1 score: %f" %(f1_score(y_test, predicted_test_labels)))