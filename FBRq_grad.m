% Implementation of the quadratic approximation of a blankout regularized logistic regression model
function [C, dC] = FBRq_grad(W,X,Y,q,lambda)
% This function expects an 1xN label vector Y with labels -1 and +1.

    % Compute negative log-likelihood
    wx = W' * X;
    m = max(wx, -wx);
    Awx = exp( wx-m) + exp(-wx-m);
    ll = -Y .* wx + log(Awx) + m;
    
    % Numerical stability issues
    q = min(q,0.9999);
    
    % Compute corrupted log-partition function
    sgm = 1./(1 + exp(-2*wx));
    Vx = bsxfun(@times, 1 ./ (1 - q) - 1, X .^ 2);
    Vy = (W .^ 2)' * Vx;
    Vwx = 4 * sgm .* (1 - sgm);
    R = .5*Vy.*Vwx;
    
    % Expected cost
    C = sum(ll+R, 2) + lambda .*sum(W.^2);

    % Only compute gradient if requested
    if nargout > 1
        
        % Compute likelihood gradient
        dll = -Y*X' + ((exp( wx-m) - exp(-wx-m))./ Awx) * X';
        
        % Compute regularizer gradient
        dR = Vwx.*((1-sgm) - sgm).*Vy*X' + W'.* (Vwx * Vx'); 
        
        % Gradient
        dC = dll' + dR' + 2.*lambda.*W;
        
    end
end
