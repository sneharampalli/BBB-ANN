function [train_err, test_err, f1_score, M] = parse_results(filename, folds)
    M = dlmread(filename, ",");
    train_err = M(:, 1);
    test_err = M(:, 2);
    f1_score = M(:, 3);
    
    % learning_rate = [1, 0.1, 0.01, 0.005, 0.001]
    % hidden_units = [4, 8, 12, 16, 21, 26, 32, 48]
    % TRAIN ERROR LOGIC
    train_err_vis = zeros(8, 5);
    [m, d] = size(train_err_vis);
    count = 40;
    for i = 1:m
        for j = 1:d
            train_err_vis(i, j) = 1 - train_err(count, 1);
            count = count - 8;
        end
        count = 40 - i;
    end
    
    figure;
    imagesc(train_err_vis);
    h = colorbar;
    title(['\fontsize{25} Training Accuracy for Different Hidden ' ...
        'Units & Learning Rates'])
    ylabel(h, 'Training Accuracy', 'FontSize', 16);
    set(gca,'FontSize', 13);
    xlabel("Learning Rates", 'FontSize', 16);
    ylabel("Number of Hidden Units", 'FontSize', 16);
    axis square;
    xticks([1 2 3 4 5]);
    xticklabels([0.001 0.005 0.01 0.1 1]);
    yticks([1 2 3 4 5 6 7 8]);
    yticklabels([4 8 12 16 21 26 32 48]);
    
    % TEST ERROR LOGIC
    test_err_vis = zeros(8, 5);
    [m, d] = size(test_err_vis);
    count = 40;
    for i = 1:m
        for j = 1:d
            test_err_vis(i, j) = 1 - test_err(count, 1);
            count = count - 8;
        end
        count = 40 - i;
    end
    
    figure;
    imagesc(test_err_vis);
    h = colorbar;
    title(['\fontsize{25} Testing Accuracy for Different Hidden ' ...
        'Units & Learning Rates'])
    ylabel(h, 'Testing Accuracy', 'FontSize', 16);
    set(gca,'FontSize', 13);
    xlabel("Learning Rates", 'FontSize', 16);
    ylabel("Number of Hidden Units", 'FontSize', 16);
    axis square;
    xticks([1 2 3 4 5]);
    xticklabels([0.001 0.005 0.01 0.1 1]);
    yticks([1 2 3 4 5 6 7 8]);
    yticklabels([4 8 12 16 21 26 32 48]);
    
    % F1-Score LOGIC
    f1_score_vis = zeros(8, 5);
    [m, d] = size(f1_score_vis);
    count = 40;
    for i = 1:m
        for j = 1:d
            f1_score_vis(i, j) = f1_score(count, 1);
            count = count - 8;
        end
        count = 40 - i;
    end
    
    figure;
    imagesc(f1_score_vis);
    h = colorbar;
    title(['\fontsize{25} F_1 Score for Different Hidden ' ...
        'Units & Learning Rates'])
    ylabel(h, 'F_1 Score', 'FontSize', 16);
    set(gca,'FontSize', 13);
    xlabel("Learning Rates", 'FontSize', 16);
    ylabel("Number of Hidden Units", 'FontSize', 16);
    axis square;
    xticks([1 2 3 4 5]);
    xticklabels([0.001 0.005 0.01 0.1 1]);
    yticks([1 2 3 4 5 6 7 8]);
    yticklabels([4 8 12 16 21 26 32 48]);
end