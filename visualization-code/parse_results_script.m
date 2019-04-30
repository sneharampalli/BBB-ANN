clc;
clear;
folds = 20;

[train_err, test_err, f1_score, M] = parse_results("cleaned_cv_results.txt", folds);
% n x m array -- n is the hidden units, m is the learning rates