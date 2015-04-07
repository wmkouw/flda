% Implementation of the quadratic approximation of the expected log-loss
function [L, dL] = far_log_dropout_grad(W,X,Y,theta,l2)
% This function expects an 1xN label vector Y with labels -1 and +1.

% Precompute
wx = W' * X;
m = 1*max(-wx,wx);
Ap = exp(wx -m) + exp(-wx -m);
An = exp(wx -m) - exp(-wx -m);
dAwx = An./Ap;
d2Awx = 2*exp(-2*m)./Ap.^2;
qX1 = bsxfun(@times, 1-theta, X);
qX2 = bsxfun(@times, -theta, X);
qX3 = bsxfun(@times, theta.*(1-theta),X.^2);
qX4 = bsxfun(@times, 2-theta,X);

% Negative log-likelihood (-log p(y|x))
L = sum(-Y.* (W'*qX1) + log(Ap) +m,2);

% First order expansion term
T1 = sum(dAwx.*(W'*qX2),2);

% Second order expansion term
Q2 = bsxfun(@times,d2Awx,qX2)*qX4' + diag(sum(bsxfun(@times,qX3,d2Awx),2));
T2 = W'*Q2*W;

% Compute loss
L = L + T1 + T2;

% Additional l2-regularization
L = L +  l2 *(sum(W(:).^2));

% Only compute gradient if requested
if nargout > 1
    
    % Compute partial derivative of negative log-likelihood
    dL = qX1*-Y' + X*dAwx';
    
    % Compute partial derivative of first-order term
    dT1 = X*((1-dAwx.^2).*(W'*qX2))' + qX2*dAwx';
    
    % Compute partial derivative of second-order term
    wQw = (W'*qX2).*(W'*qX4) + W'.^2*qX3;
    dT2 = (Q2+Q2')*W + X*(-4*exp(-2*m).*An./Ap.^3.*wQw)';
    
    % Gradient
    dL = dL + dT1 + dT2;
    
    % Additional l2-regularization
    dL = dL + 2.*l2.*W(:);
end
end
