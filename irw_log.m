function [W] = irw_log(X,y,theta,l2)
% Function to start optimize an instance reweighted loss function

addpath(genpath('minFunc'));

options.DerivativeCheck = 'off';
options.Display = 'valid';
options.Method = 'lbfgs';

% Check for y in {1,..K}
y(y== 0) = 2;
y(y==-1) = 2;

% Shape
[M,~] = size(X);
K = numel(unique(y));

% Minimize loss
W_star = minFunc(@mWLR_grad, zeros(M*K,1), options, X(1:end-1,:), y, theta, l2);
    
% Output multiclass weight vector
W = [reshape(W_star(1:end-K), [(M-1) K]); W_star(end-K+1:end)'];

end
