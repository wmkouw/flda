function [pred,G] = gfk(X,Z,yX,varargin)
% Implementation of a Geodesic Flow Kernel for Domain Adaptation
%
% ref: Geodesic Flow Kernel for Unsupervised Domain Adaptation.
% B. Gong, Y. Shi, F. Sha, and K. Grauman.
% Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Providence, RI, June 2012.

% Parse input
p = inputParser;
addParameter(p, 'nE', []);
addParameter(p, 'l2', 1e-3);
addParameter(p, 'clf', '1-nn')
parse(p, varargin{:});

% Check sizes
[D,NZ]=  size(Z);

% Preprocess data
X = da_prep(X, {'sum_samp', 'zscore'});
Z = da_prep(Z, {'sum_samp', 'zscore'});

if isempty(p.Results.nE)
    
    % Find principal components
    [PX,~] = pca(full(X'), 'Economy', false);
    [PZ,~] = pca(full(Z'), 'Economy', false);
    [PP,~] = pca(full([X'; Z']), 'Economy', false);

    % Subspace disagreement measure    
    alpha = acos(diag(PX'*PP));
    beta = acos(diag(PZ'*PP));
    SDM = 0.5*(sin(alpha) + sin(beta));
    dix = find(SDM>0.999,1,'first');
    dix = min(dix,floor(D./2));
    disp(['Optimal dimensionality (SDM): ' num2str(dix)]);
else
    dix = min(p.Results.nE,floor(D./2));
    
    % Find principal components
    [PX,~] = pca(full(X'), 'Economy', false, 'NumComponents', dix);
    [PZ,~] = pca(full(Z'), 'Economy', false, 'NumComponents', dix);
end

% Find geodesic flow kernel
G = GFK([PX,null(PX')], PZ(:,1:dix));

% Perform classification
switch p.Results.clf
    case {'1nn', '1-nn'}
        [pred] = k_1nn(G, X', yX', Z');
    case {'lr', 'log'}
        W = mLR(X'*G,yX,'l2',p.Results.l2);
        [~,pred] = max([Z'*G ones(NZ,1)]*W,[],2);
    case {'svm'}
        W = svmtrain(yX,X'*G);
        pred = svmpredict(zeros(yZ),Z'*G,W);
end

end

function G = GFK(Q,Pt)
% Input: Q = [Ps, null(Ps')], where Ps is the source subspace, column-wise orthonormal
%        Pt: target subsapce, column-wise orthonormal, D-by-d, d < 0.5*D
% Output: G = \int_{0}^1 \Phi(t)\Phi(t)' dt

% ref: Geodesic Flow Kernel for Unsupervised Domain Adaptation.
% B. Gong, Y. Shi, F. Sha, and K. Grauman.
% Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Providence, RI, June 2012.

% Contact: Boqing Gong (boqinggo@usc.edu)

N = size(Q,2); %
dim = size(Pt,2);

% compute the principal angles
QPt = Q' * Pt;
[V1,V2,V,Gam,Sig] = gsvd(QPt(1:dim,:), QPt(dim+1:end,:));
V2 = -V2;
theta = real(acos(diag(Gam))); % theta is real in theory. Imaginary part is due to the computation issue.

% compute the geodesic flow kernel
eps = 1e-20;
B1 = 0.5.*diag(1+sin(2*theta)./2./max(theta,eps));
B2 = 0.5.*diag((-1+cos(2*theta))./2./max(theta,eps));
B3 = B2;
B4 = 0.5.*diag(1-sin(2*theta)./2./max(theta,eps));
G = Q * [V1, zeros(dim,N-dim); zeros(N-dim,dim), V2] ...
    * [B1,B2,zeros(dim,N-2*dim);B3,B4,zeros(dim,N-2*dim);zeros(N-2*dim,N)]...
    * [V1, zeros(dim,N-dim); zeros(N-dim,dim), V2]' * Q';

end

function [pred] = k_1nn(M, X, y, Z)

% Distance according to metric M
D = repmat(diag(X*M*X'),1,size(Z,1)) ...
    + repmat(diag(Z*M*Z')',size(X,1),1)...
    - 2*X*M*Z';

% Find minimal distance
[~, ix] = min(D);

% Predict according to nearest neighbour
pred = y(ix);

end

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
options.Display = 'valid';
options.DerivativeCheck = 'off';
options.maxIter = 1e4;
options.xTol = 1e-5;
options.Method = 'lbfgs';

% Check for bias augmentation
if ~all(X(:,end)==1); X = [X ones(size(X,1),1)]; end

% Shape
[N,D] = size(X);

% Number of classes
K = numel(unique(y));

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

function [L, dL] = mLR_grad(B,X,y,lambda)
% Logistic regression gradient
% Wouter Kouw
% 29-09-2014

% Shape
[~,D] = size(X);
K = numel(unique(y));
B = reshape(B, [D K]);

% Numerical stability
XB = X*B;
a = max(XB, [], 2);
XB = bsxfun(@minus, XB, a);

% Point-wise weighted negative log-likelihood
L = -sum(X.*B(:,y)',2) + log(sum(exp(XB),2)) + a;

% Add l2-regularizer
L = sum(L,1) + lambda .* sum(B(:).^2);

% Only compute gradient if requested
if nargout > 1
    
    % Gradient with respect to B_k
    dL = zeros(D,K);
    for k = 1:K
        dL(:,k) = sum( -bsxfun(@times,(y==k),X)' + bsxfun(@times,exp(XB(:,k))./sum(exp(XB),2),X)',2);
    end
    
    % Add l2-regularizer
    dL = dL(:) + 2*lambda.*B(:);
    
end
end
