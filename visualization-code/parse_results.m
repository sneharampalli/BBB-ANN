function [train_acc, test_acc, f1_score, M] = parse_results(filename, folds)
    M = dlmread(filename, ",");
    train_acc = M(:, 1);
    test_acc = M(:, 2);
    f1_score = M(:, 3);
    
    % TRAIN ERROR LOGIC
    train_acc_vis = zeros(10, 8);
    [m, d] = size(train_acc_vis);
    count = m * d;
    for i = 1:m
        for j = 1:d
            train_acc_vis(i, j) = train_acc(count, 1);
            count = count - 8;
        end
        count = 80 - i;
    end
    
    figure;
    imagesc(train_acc_vis);
    h = colorbar;
    title(['\fontsize{25} Training Accuracy for Different Numbers of Hidden ' ...
        'Units & Learning Rates'])
    ylabel(h, 'Training Accuracy', 'FontSize', 16);
    set(gca,'FontSize', 13);
    xlabel("Learning Rates", 'FontSize', 16);
    ylabel("Number of Hidden Units", 'FontSize', 16);
    axis square;
    
    xticks([1 2 3 4 5 6 7 8]);
    xticklabels([0.1, 0.01, 0.005, 0.5, 1, 2, 5, 10]);
    yticks([1 2 3 4 5 6 7 8 9 10]);
    yticklabels([4, 8, 12, 16, 21, 26, 32, 48, 54, 60]);
    
    % TEST ERROR LOGIC
    test_acc_vis = zeros(10, 8);
    [m, d] = size(test_acc_vis);
    count = m * d;
    for i = 1:m
        for j = 1:d
            test_acc_vis(i, j) = test_acc(count, 1);
            count = count - 8;
        end
        count = 80 - i;
    end
    
    figure;
    imagesc(test_acc_vis);
    h = colorbar;
    title(['\fontsize{25} Testing Accuracy for Different Numbers of Hidden ' ...
        'Units & Learning Rates'])
    ylabel(h, 'Testing Accuracy', 'FontSize', 16);
    set(gca,'FontSize', 13);
    xlabel("Learning Rates", 'FontSize', 16);
    ylabel("Number of Hidden Units", 'FontSize', 16);
    axis square;
    
    xticks([1 2 3 4 5 6 7 8]);
    xticklabels([0.1, 0.01, 0.005, 0.5, 1, 2, 5, 10]);
    yticks([1 2 3 4 5 6 7 8 9 10]);
    yticklabels([4, 8, 12, 16, 21, 26, 32, 48, 54, 60]);

    % F1-Score LOGIC
    f1_score_vis = zeros(10, 8);
    [m, d] = size(f1_score_vis);
    count = m * d;
    for i = 1:m
        for j = 1:d
            f1_score_vis(i, j) = f1_score(count, 1);
            count = count - 8;
        end
        count = 80 - i;
    end
    
    figure;
    imagesc(f1_score_vis);
    h = colorbar;
    title(['\fontsize{25} F_1 Score for Different Numbers of Hidden ' ...
        'Units & Learning Rates'])
    ylabel(h, 'F_1 Score', 'FontSize', 16);
    set(gca,'FontSize', 13);
    xlabel("Learning Rates", 'FontSize', 16);
    ylabel("Number of Hidden Units", 'FontSize', 16);
    axis square;
    
    xticks([1 2 3 4 5 6 7 8]);
    xticklabels([0.1, 0.01, 0.005, 0.5, 1, 2, 5, 10]);
    yticks([1 2 3 4 5 6 7 8 9 10]);
    yticklabels([4, 8, 12, 16, 21, 26, 32, 48, 54, 60]);
end