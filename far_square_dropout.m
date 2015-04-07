function [w] = far_square_dropout(X,y,theta, lambda)
% Function to calculate feature absence regularized least squares classifier
% Note: no optimization is necessary for ls

% First two moments of transfer distribution
EX = bsxfun(@times, (1-theta), X);
VX = diag(theta.*(1-theta)).*(X*X');

% Least squares solution
w = (EX*EX' + VX + lambda*eye(size(X,1)))\EX*y;

end