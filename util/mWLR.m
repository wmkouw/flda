function [B] = mWLR(X,y,varargin)
% Importance Weighted Logistic regression
% Input:
%    X is in NxD format
%    y is label vector in [1,..K]
% Optional:
%    l2: regularization parameter
%    iw: importance weights
% Output:
%    W is DxK resulting classifier
%
% Wouter Kouw
% 15-09-2014

% Dependencies
addpath(genpath('minFunc'));

% Parse input
p = inputParser;
addParameter(p, 'iw', ones(1,size(X,1)));
addParameter(p, 'l2', 1e-3);
parse(p, varargin{:});

% Optimization options
options.Display = 'final';
options.DerivativeCheck = 'off';
options.maxIter = 1e4;
options.xTol = 1e-5;
options.Method = 'lbfgs';

% Check for bias augmentation
if ~all(X(:,end)==1); X = [X ones(size(X,1),1)]; end

% Shape
[N,D] = size(X);

% Number of classes
K = max(max(y),numel(unique(y))); 

% Check column vector y, iw
if size(y,1)~=N; y = y'; end
if size(p.Results.iw,1)~=N; iw = p.Results.iw'; else iw = p.Results.iw; end

% Check for y in {1,..K}
y(y == 0) = 2;
y(y ==-1) = 2;

% Minimize loss
B = minFunc(@mWLR_grad, rand(D*K,1), options, X,y, iw, p.Results.l2);

% Reshape into K-class matrix
B = reshape(B, [D K]);

end

function [L, dL] = mWLR_grad(B,X,y,iw,lambda)
% Importance Weighted Logistic Regression gradient
% Wouter Kouw
% 29-09-2014

% Shape
[~,D] = size(X);
K = max(max(y),numel(unique(y))); 
B = reshape(B, [D K]);

% Numerical stability
XB = X*B;
a = max(XB, [], 2);
XB = bsxfun(@minus, XB, a);

% Point-wise weighted negative log-likelihood
L = -iw.*sum(X.*B(:,y)',2) + iw.*log(sum(exp(XB),2)) + iw.*a;

% Add l2-regularizer
L = sum(L,1) + lambda .* sum(B(:).^2);

% Only compute gradient if requested
if nargout > 1
  
    % Gradient with respect to B_k
    dL = zeros(D,K);
    for k = 1:K
        dL(:,k) = sum( -bsxfun(@times,iw.*(y==k),X)' + bsxfun(@times,iw.*exp(XB(:,k))./sum(exp(XB),2),X)',2);
    end
    
	% Add l2-regularizer
	dL = dL(:) + 2*lambda.*B(:);
    
end
end
