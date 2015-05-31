clear; clc; close all;

load('dataset.mat');

params = struct( ...
    'kernel', 'rbf', ...
    'BASIS_SIZE', 200, ...
    'MAXEPOCHS',20, ...
    'gamma', 0.004, ...
    'theta', 1, ...
    'd', 2);

%SLACKMIN
NUM_FOLDS = 10;
PERCENT_OUT = 0.2;
P = size(x,2);
accuracy_train = zeros(1, NUM_FOLDS);
accuracy_test = zeros(1, NUM_FOLDS);
accuracySVM_train = zeros(1, NUM_FOLDS);
accuracySVM_test = zeros(1, NUM_FOLDS);
timeSlackmin = zeros(1, NUM_FOLDS);
timeSVM = zeros(1, NUM_FOLDS);
for fold = 1:NUM_FOLDS
    fprintf('Fold #%-2d\n', fold);
    [train_idx, test_idx] = crossvalind('HoldOut', P, PERCENT_OUT);
    % Train model
    timeStart = tic;
    [model, y, accuracy_train(fold)] = slackmin_train(x(:,train_idx), t(train_idx), params);
    timeSlackmin(fold) = toc(timeStart);% Test model
    [y_test, accuracy_test(fold)] = slackmin_sim(x(:,test_idx), t(test_idx), model);
    
    
    %SVM
    % -t kernel_type : set type of kernel function (default 2)
    % 	0 -- linear: u'*v
    % 	1 -- polynomial: (gamma*u'*v + coef0)^degree
    % 	2 -- radial basis function: exp(-gamma*|u-v|^2)
    % 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
    % 	4 -- precomputed kernel (kernel values in training_instance_matrix)
    % -d degree : set degree in kernel function (default 3)
    % -g gamma : set gamma in kernel function (default 1/num_features)
    % -r coef0 : set coef0 in kernel function (default 0)
    % -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
    timeStart = tic;
    modelSVM = svmtrain(t(train_idx)', x(:,train_idx)', '-t 2 -g 0.004 -c 10000');
    timeSVM(fold) = toc(timeStart);
    [~, vec, ~] = svmpredict(t(train_idx)', x(:,train_idx)', modelSVM);
    accuracySVM_train(fold) = vec(1);
    [~, vec, ~] = svmpredict(t(test_idx)', x(:,test_idx)', modelSVM);
    accuracySVM_test(fold) = vec(1);
end
fprintf('\n***** OVERALL RESULTS *****\n\n');
fprintf('>>>> Slackmin: Mean Train accuracy = %-0.2f\n', mean(accuracy_train));
fprintf('>>>> Slackmin: Mean Test accuracy = %-0.2f\n', mean(accuracy_test));
fprintf('>>>> Slackmin: Mean Training time = %-0.2f  (sec)\n', mean(timeSlackmin));
fprintf('++++ SVM: Mean Train accuracy = %-0.2f\n', mean(accuracySVM_train));
fprintf('++++ SVM: Mean Test accuracy = %-0.2f\n', mean(accuracySVM_test));
fprintf('++++ SVM: Mean Training time = %-0.2f (sec)\n', mean(timeSVM));



