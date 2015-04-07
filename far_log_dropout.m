function [w] = far_log_dropout(X,y,theta, l2)
% Function to run the optimization procedure of the feature absence
% regularized domain adaptation classifier.

addpath(genpath('minFunc'));

% Shape
[M,N] = size(X);

options.DerivativeCheck = 'off';
options.Method = 'lbfgs';
options.Display = 'valid';

% Analytical solution to theta=1 => w=0
ix = find(theta~=1);

% Minimize loss
w_star = minFunc(@far_log_dropout_grad, zeros(length(ix),1), options, X(ix,:),y,theta(ix),l2);

% Bookkeeping
w = zeros(M,1);
w(ix) = w_star;

end