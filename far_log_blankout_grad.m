% Implementation of the quadratic approximation of the expected log-loss
function [L, dL] = far_log_blankout_grad(W,X,Y,theta,l2)
% This function expects an 1xN label vector Y with labels -1 and +1.

    % Compute negative log-likelihood
    wx = W' * X;
    m = max(wx, -wx);
    Awx = exp( wx-m) + exp(-wx-m);
    ll = -Y .* wx + log(Awx) + m;
    
    % Numerical stability issues
    theta = min(theta,0.9999);
    
    % Compute corrupted log-partition function
    sgm = 1./(1 + exp(-2*wx));
    Vx = bsxfun(@times, 1 ./ (1 - theta) - 1, X .^ 2);
    Vy = (W .^ 2)' * Vx;
    Vwx = 4 * sgm .* (1 - sgm);
    R = .5*Vy.*Vwx;
    
    % Expected cost
    L = sum(ll+R, 2) + l2 .*sum(W.^2);

    % Only compute gradient if requested
    if nargout > 1
        
        % Compute likelihood gradient
        dll = -Y*X' + ((exp( wx-m) - exp(-wx-m))./ Awx) * X';
        
        % Compute regularizer gradient
        dR = Vwx.*((1-sgm) - sgm).*Vy*X' + W'.* (Vwx * Vx'); 
        
        % Gradient
        dL = dll' + dR' + 2.*l2.*W;
        
    end
end
