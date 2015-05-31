% Slackmin recall
% Kostas Diamantaras, May 2015
% usage: [y, accuracy] = slackmin_sim(x, t, model)
%   x = [nxP] pattern matrix
%       (n = pattern dimension, P = number of patterns)
%   t = [1xP] target vector (values = -1/1)
%   model = srtuct containing model parameters created by slackmin_train()
%
% Returns:
%   y = output vector (values = double)
%       ideally (y>0) if t=+1,  (y<0) if t=-1
%   accuracy = classification accuracy (value = double between 0 and 100)
%
function [y, accuracy] = slackmin_sim(x, t, model)

[n,P] = size(x);
[n1,Pb] = size(model.xb);
[nt,Pt] = size(t);

if n1 ~= n
    fprintf(2, 'slackmin_sim: basis patterns must have same dimension as test patterns\n');
    return;
end
if P ~= Pt
    fprintf(2, 'slackmin_sim: number of patterns must match number of targets\n');
    return;
end
if nt ~= 1
    fprintf(2, 'slackmin_sim: each pattern must have one target\n');
    return;
end



switch model.kernel
    case 'linear'
        K = [x; ones(1,P)];
    case 'rbf'
        B = model.BASIS_SIZE;
        gamma = model.gamma;
        K = zeros(B,P);
        for i=1:B
            for j=1:P
                K(i,j) = exp( -gamma * norm(model.xb(:,i) - x(:,j))^2 );
            end
        end
    case 'poly'
        B = model.BASIS_SIZE;
        theta = model.theta;
        d = model.d;
        K = zeros(B,P);
        for i=1:B
            for j=1:P
                K(i,j) = ( model.xb(:,i)' * x(:,j) + theta )^d;
            end
        end
end

y = model.w' * K;
accuracy = 100 * sum((t.*y)>0) / P;
fprintf('  [slackmin_sim] Accuracy = %0.3f%%\n', accuracy);