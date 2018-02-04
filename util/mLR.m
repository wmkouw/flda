function [B] = mLR(X,y,varargin)
% Logistic regression
% Input:
%    X is in NxD format
%    y is label vector in [1,..K]
% Output:
%    W is DxK resulting classifier
%
% Wouter Kouw
% 15-09-2014

% Dependencies
addpath(genpath('minFunc'));

% Parse input
p = inputParser;
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

% Check column vector y
if size(y,1)~=N; y = y'; end

% Check for y in {1,..K}
y(y == 0) = 2;
y(y ==-1) = 2;

% Minimize loss
B = minFunc(@mLR_grad, rand(D*K,1), options, X,y, p.Results.l2);

% Reshape into K-class matrix
B = reshape(B, [D K]);

end

function [L, dL] = mLR_grad(B,X,y,l2)
% Logistic regression gradient
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

% Convenient variables
eXB = exp(XB);
eA = sum(eXB,2);

% Point-wise weighted negative log-likelihood
ll = -sum(X.*B(:,y)',2) + log(eA) + a;

% Add l2-regularizer
L = sum(ll,1) + l2.*sum(B(:).^2);

% Only compute gradient if requested
if nargout > 1
  
    % Gradient with respect to B_k
    dll = zeros(D,K);
    for k = 1:K
        dll(:,k) = -X'*(y==k) + X'*(eXB(:,k)./eA);
    end
    
	% Add l2-regularizer
	dL = dll(:) + 2*l2.*B(:);
    
end
end
