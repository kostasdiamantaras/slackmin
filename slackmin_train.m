% Slackmin training algorithm
% Kostas Diamantaras, May 2015
% usage: [model, y, accuracy] = slackmin_train(x, t, params)
%   x = [nxP] pattern matrix
%       (n = pattern dimension, P = number of patterns)
%   t = [1xP] target vector (values = -1/1)
%   params = struct(
%      'kernel', (values = 'linear'(default), 'rbf', 'poly'), ...
%      'BASIS_SIZE', (values = interger between 1 and P, default = min(100,P)), ...
%      'MAXEPOCHS', (values = number of training epochs, default = 20), ...
%       % parameter gamma, for RBF kernel  K(x,y) = exp(-gamma * norm(x-y)^2):
%      'gamma', (values = positive double, default = 1), ...
%       % parameters theta, d, for Polynomial kernel  K(x,y) = (x'*y + theta)^d:
%      'theta', (values = double, default = 1), ...
%      'd', (values = integer, default = 2) ...
%   )
%
% Returns:
%   y = output vector (values = double)
%       ideally (y>0) if t=+1,  (y<0) if t=-1
%   accuracy = classification accuracy (value = double between 0 and 100)
%
function [model, y, accuracy] = slackmin_train(x, t, params)


SHOW_PROGRESS = 0;
[n,P] = size(x);
[nt,Pt] = size(t);
if P ~= Pt
    fprintf(2, 'slackmin_train: number of patterns must match number of targets\n');
    return;
end
if nt ~= 1
    fprintf(2, 'slackmin_train: each pattern must have one target\n');
    return;
end

%%%%%%%%%% CHECK params %%%%%%%%%%
try
    exist(params.kernel, 'var');
catch
    params.kernel = 'linear';
end
try
    exist(params.BASIS_SIZE, 'var');
catch
    params.BASIS_SIZE = min(100, P);
end
try
    exist(params.MAXEPOCHS, 'var');
catch
    params.MAXEPOCHS = 20;
end
if strcmp(params.kernel, 'rbf')
    try
        exist(params.gamma, 'var');
    catch
        params.gamma = 1;
    end
end
if strcmp(params.kernel, 'poly')
    try
        exist(params.theta, 'var');
    catch
        params.theta = 1;
    end
    try
        exist(params.d, 'var');
    catch
        params.d = 2;
    end
end


switch params.kernel
    case 'linear'
        w = randn(n+1,1); % weight vector
        K = [x; ones(1,P)];
        BASIS = [];
        gamma = [];
        theta = [];
        d = [];
    case 'rbf'
        B = params.BASIS_SIZE;
        gamma = params.gamma;
        theta = [];
        d = [];
        BASIS = randperm(P, B);
        w = randn(B,1); % weight vector
        K = zeros(B,P);
        for i=1:B
            for j=1:P
                K(i,j) = exp( -gamma * norm(x(:,BASIS(i)) - x(:,j))^2 );
            end
        end
    case 'poly'
        B = params.BASIS_SIZE;
        gamma = [];
        theta = params.theta;
        d = params.d;
        BASIS = randperm(P, B);
        w = randn(B,1); % weight vector
        K = zeros(B,P);
        for i=1:B
            for j=1:P
                K(i,j) = ( x(:,BASIS(i))' * x(:,j) + theta )^d;
            end
        end
end

%%%%%%%%%% LEARNING ALGORITHM %%%%%%%%%%
MAXEPOCHS = params.MAXEPOCHS;
misclass_best = inf;
if ~SHOW_PROGRESS
    fprintf('  [slackmin_train] ');
end
for epoch = 1:MAXEPOCHS
    y = w' * K;
    subidx = find((t.*y) < 1);
    num_sub = length(subidx);
    misclass = sum((t.*y) < 0);
    % Show error progress
    if SHOW_PROGRESS
        fprintf('  [slackmin_train] epoch %-4d: Sub-1 vectors %5d (%0.3f%%), Accuracy = %0.3f%%\n', ...
            epoch, num_sub, num_sub/P*100, (1-misclass/P)*100);
    else
        fprintf('.');
    end
    % Keep the best performer so far
    if misclass < misclass_best
        w_best = w;
        subidx_best = subidx;
        num_sub_best = num_sub;
        misclass_best = misclass;
    end
    % Update weights
    w = pinv(K(:,subidx)') * t(subidx)';
end


%%%%%%%%%% RESULTS %%%%%%%%%%
accuracy = 100*(1 - misclass_best/P);
fprintf('\n  [slackmin_train] Sub-1 vectors %5d (%0.3f%%), Accuracy = %0.3f%%\n', ...
    num_sub_best, num_sub_best/P*100, accuracy);

model = struct( ...
    'kernel', params.kernel, ...
    'BASIS_SIZE', length(BASIS), ...
    'xb', x(:,BASIS), ...
    'w', w_best, ...
    'subidx', subidx_best, ...
    'gamma', gamma, ...
    'theta', theta, ...
    'd', d);