function [L, dL] = irw_log_grad(W,X,y,iw, lambda)
% Implementation of weighted logistic regression
% Wouter Kouw
% 29-09-2014
% This function expects an 1xN label vector y with labels [1,..,K] and a
% weight vector of MxK+K

% Shape
[M,N] = size(X);
K = numel(unique(y));
W0 = reshape(W(M*K+1:end), [1 K]);
W = reshape(W(1:M*K), [M K]);

% Compute p(y|x)
WX = bsxfun(@plus, W' * X, W0');
WX = exp(bsxfun(@minus, WX, max(WX, [], 1)));
WX = bsxfun(@rdivide, WX, max(sum(WX, 1), realmin));

% Negative log-likelihood of each sample
L = 0;
for i=1:N
    L = L - log(max(iw(i)*WX(y(i), i), realmin));
end
L = L + lambda .* sum([W(:); W0(:)] .^ 2);

% Only compute gradient if requested
if nargout > 1
    
    % Compute positive part of gradient
	pos_E = zeros(M, K);
    pos_E0 = zeros(1, K);
    for k=1:K
        pos_E(:,k) = sum(bsxfun(@times, iw(y == k), X(:,y == k)), 2);            
    end
    for k=1:K
        pos_E0(k) = sum(y == k);
    end
    
    % Compute negative part of gradient    
    neg_E = bsxfun(@times, iw, X) * WX';
    neg_E0 = sum(WX, 2)';
        
	% Compute gradient
	dL = -[pos_E(:) - neg_E(:); pos_E0(:) - neg_E0(:)] + 2 .* lambda .* [W(:); W0(:)];
    
end
end
