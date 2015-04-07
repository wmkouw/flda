function [W] = irw_log(X,y,theta,l2)
% Function to start optimize an instance reweighted loss function

options.DerivativeCheck = 'off';
options.Display = 'valid';
options.Method = 'lbfgs';

% Shape
[M,~] = size(X);
K = numel(unique(y));
if K==2
    % Convert binary labels to multi-class    
    y(y==-1) = 2;
    W_star = minFunc(@mWLR_grad, zeros(M*K,1), options, X(1:end-1,:), y, theta, l2);
    
    % Output single weight vector
    W = [W_star(1:M-1); W_star(end-1)];
else
    
    W_star = minFunc(@mWLR_grad, zeros(M*K,1), options, X(1:end-1,:), y, theta, l2);
    
    % Output multiclass weight vector
    W = [reshape(W_star(1:end-K), [(M-1) K]); W_star(end-K+1:end)'];
end

end
