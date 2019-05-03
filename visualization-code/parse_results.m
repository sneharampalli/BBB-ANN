function [train_acc, test_acc, f1_score, M] = parse_results(filename, type_data, num_rows, num_cols)
    % type_data = 'Diagnostic' or 'Original'
    M = dlmread(filename, ",");
    train_acc = M(:, 1);
    test_acc = M(:, 2);
    f1_score = M(:, 3);
    
    % BEST TEST ACCURACY
    [best_test_acc, index] = max(test_acc);
    [next_test_acc, index2] = max(test_acc(test_acc~=max(test_acc)));
    disp("NEXT BEST");
    disp(next_test_acc);
    disp(index2);
    disp("BEST TEST ACCURACY");
    disp(best_test_acc);
    disp("CORRESPONDING TRAIN ACCURACY");
    disp(train_acc(index, 1));
    disp("CORRESPONDING F1 Score");
    disp(f1_score(index, 1));
    disp("INDEX");
    disp(index);
    
    % BEST TRAIN ACCURACY
    [~, index] = max(train_acc);
    disp("INDEX");
    disp(index);
    [~, index2] = max(train_acc(train_acc~=max(train_acc)));
    disp("NEXT BEST");
    disp(index2);
    
    % BEST F1 SCORE
    [~, index] = max(f1_score);
    disp("INDEX");
    disp(index);
    [~, index2] = max(f1_score(f1_score~=max(f1_score)));
    disp("NEXT BEST");
    disp(index2);
    
    % TRAIN ERROR LOGIC
    train_acc_vis = zeros(num_rows, num_cols);
    [m, d] = size(train_acc_vis);
    count = 1;
    for j = 1:d
        for i = 1:m
            a = m - i + 1;
            train_acc_vis(a, j) = train_acc(count, 1);
            count = count + 1;
        end
    end
    
    figure;
    subplot(1, 3, 1);
    imagesc(train_acc_vis);
    h = colorbar;
    title({'\fontsize{19} Training Accuracies for ', strcat(type_data, ' Dataset')})
    ylabel(h, 'Training Accuracy', 'FontSize', 14);
    set(gca,'FontSize', 14);
    xlabel("Learning Rate", 'FontSize', 14);
    ylabel("Number of Hidden Units", 'FontSize', 14);
    axis square;
    
    xticks([1 2 3 4 5 6 7 8]);
    xticklabels([0.005, 0.01, 0.1, 0.5, 1, 2, 5, 10]);
    yticks([1 2 3 4 5 6 7 8 9 10 11 12]);
    yticklabels([60, 54, 48, 32, 26, 21, 16, 14, 12, 10, 8, 4]);

    % TEST ERROR LOGIC
    test_acc_vis = zeros(num_rows, num_cols);
    [m, d] = size(test_acc_vis);
    count = 1;
    for j = 1:d
        for i = 1:m
            a = m - i + 1;
            test_acc_vis(a, j) = test_acc(count, 1);
            count = count + 1;
        end
    end
    subplot(1, 3, 2);
    imagesc(test_acc_vis);
    h = colorbar;
    title({'\fontsize{19} Testing Accuracies for ', strcat(type_data, ' Dataset')})
    ylabel(h, 'Testing Accuracy', 'FontSize', 14);
    set(gca,'FontSize', 14);
    xlabel("Learning Rate", 'FontSize', 14);
    ylabel("Number of Hidden Units", 'FontSize', 14);
    axis square;
    
    xticks([1 2 3 4 5 6 7 8]);
    xticklabels([0.005, 0.01, 0.1, 0.5, 1, 2, 5, 10]);
    yticks([1 2 3 4 5 6 7 8 9 10 11 12]);
    yticklabels([60, 54, 48, 32, 26, 21, 16, 14, 12, 10, 8, 4]);

    % F1-Score LOGIC
    f1_score_vis = zeros(num_rows, num_cols);
    [m, d] = size(f1_score_vis);
    count = 1;
    for j = 1:d
        for i = 1:m
            a = m - i + 1;
            f1_score_vis(a, j) = f1_score(count, 1);
            count = count + 1;
        end
    end
    
    subplot(1, 3, 3);
    imagesc(f1_score_vis);
    h = colorbar;
    title({'\fontsize{19} F_1 Scores for ', strcat(type_data, ' Dataset')})
    ylabel(h, 'F_1 Score', 'FontSize', 14);
    set(gca,'FontSize', 14);
    xlabel("Learning Rate", 'FontSize', 14);
    ylabel("Number of Hidden Units", 'FontSize', 14);
    axis square;
    
    xticks([1 2 3 4 5 6 7 8]);
    xticklabels([0.005, 0.01, 0.1, 0.5, 1, 2, 5, 10]);
    yticks([1 2 3 4 5 6 7 8 9 10 11 12]);
    yticklabels([60, 54, 48, 32, 26, 21, 16, 14, 12, 10, 8, 4]);
end