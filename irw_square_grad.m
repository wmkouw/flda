function [L, dL] = irw_square_grad(W,X,Y,iw,l2)
% Implementation of multi-class weighted least squares
% Wouter Kouw
% 15-09-2014
% This function expects an KxN label vector y with labels 0 and +1.

    % Get dimensionality and number of classes
    W = reshape(W, [3 3]);
    [M,K] = size(W);
    
    % Expectation of the negative log-likelihood
    bX = [bsxfun(@times, iw, X(1:M-1,:)); X(end,:)];
    
    % 1 vs all for all classes
    L = sum(sum((Y - W'*bX).^2,2) + l2*sum(w'.^2,1),1);

    % Only compute gradient if requested
    if nargout > 1
        
        % Gradient w.r.t. w
        dL = (Y*bX'/ (bX*bX'+l2*eye(M)))';
        dL = dL(:);
        
    end
end
