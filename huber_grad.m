function [L,dL] = huber_grad(w,X,y,la)
% Modified Huber loss function 
% R. Ando & T. Zhang (2005a). A framework for learning predictive
% structures from multiple tasks and unlabeled data. JMLR.

% Precompute
Xy = bsxfun(@times, X, y);
wXy = w'*Xy;

% Indices of discontinuity
ix = (wXy>=-1);

% Loss
L = sum(max(0,1-wXy(ix)).^2,2) + sum(-4*wXy(~ix),2);
dL = sum(bsxfun(@times, 2*max(0,1-wXy(ix)), (-Xy(:,ix))),2) + sum(-4*Xy(:,~ix),2);
    
% Add l2-regularization
L = L + la*sum(w.^2);
dL = dL + 2*la*w;

end