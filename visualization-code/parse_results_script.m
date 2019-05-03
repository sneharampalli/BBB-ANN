clc;
clear;
folds = 20;

[train_err, test_err, f1_score, M] = parse_results('diagnostic_results_clean.txt', ' Original', 12, 8);
% [train_err, test_err, f1_score, M] = parse_results("new_clean_results.txt", ' Diagnostic', 12, 8);
% n x m array -- n is the hidden units, m is the learning rates