function [W,C,pred,varargout] = tca(X,Z,yX,yZ,varargin)
% Function to perform Transfer Component Analysis.
% Pan, Tsang, Kwok, Yang (2009). Domain Adaptation via Transfer Component Analysis.
% Uses code shared by prof. Pan
%
% Applies a support vector machine
%
% Input:    X       source data (D features x N samples)
%           Z       target data (D features x M samples)
%           yX      source labels (N x 1)
% Optional:
%           yZ      target labels
%           V       validation data
%           dim     Number of transfer components (default: 10)
%           ktype   Type of kernel (default: rbf)
%           kparam  Parameter for kernel (default: 1)
%           mu      trade-off parameter transfer components (default: 1)
%           l2      l2-regularization parameters (default: 1e-3)
%
% Output:   W       Classifier parameters
%           TX      Source data mapped onto transfer components
%           TZ      Target data mapped onto transfer components
%           TV      Validation data mapped onto transfer components	
%           pred    target label predictions
% Optional:
%           {1}     Error of target label predictions
%
% Copyright: Wouter M. Kouw
% Last update: 10-05-2016

% Parse input
p = inputParser;
addParameter(p, 'yZ', []);
addParameter(p, 'V', []);
addParameter(p, 'yV', []);
addParameter(p, 'nE', []);
addParameter(p, 'Kt', 'rbf');
addParameter(p, 'kP', 1);
addParameter(p, 'gamma', 1);
addParameter(p, 'mu', 1);
addParameter(p, 'l2', 0);
parse(p, varargin{:});

% Shapes
[~,NX] = size(X);
[~,NZ] = size(Z);

% Parse input
p = inputParser;
addParameter(p, 'gamma', 1);
addParameter(p, 'm', 100);
addParameter(p, 'l2', 0);
addParameter(p, 'mu', 1);
addParameter(p, 'theta', 1);
parse(p, varargin{:});

% Find components
[C,K] = tc(X, Z, 'theta', p.Results.theta, 'mu', p.Results.mu, 'm', p.Results.m);

% Train gamma optimized svm
W = svmtrain(yX, K(1:NX,:)'*C, ['-c ' num2str(p.Results.l2) ' -t 2 -q -g ' num2str(gamma(gix))]);

% Do classification on target set
if ~isempty(p.Results.yZ);
	% Make predictions
	[pred,acc,~] = svmpredict(p.Results.yZ, [K(NX+1:NX+NZ,:)'*M; ones(1,NZ)], W);
	% Compute classification error
    varargout{1} = (100-acc(1))./100;
else
	% Make predictions on given target samples
    [pred] = svmpredict(zeros(NZ,1), [K(NX+1:NX+NZ,:)'*M; ones(1,NZ)], W, []);
end

end


function [M,K] = tc(X,Z,varargin)
% At the moment, only a radial basis function kernel implemented

% Parse input
p = inputParser;
addParameter(p, 'theta', 1);
addParameter(p, 'mu', 1);
addParameter(p, 'm', 100);
parse(p, varargin{:});

% Shapes
[~,NX] = size(X);
[~,NZ] = size(Z);

% Form block kernels
K = rbf_kernel(X,Z, 'theta', p.Results.theta);
clear X Z

% Objective function
[M,~] = eigs((eye(NX+NZ)+p.Results.mu*K*[ones(NX)./NX.^2 -ones(NX,NZ)./(NX*NZ); ...
    -ones(NZ,NX)./(NX*NZ) ones(NZ)./NZ.^2]*K)\(K*((1-1./(NX+NZ)).*eye(NX+NZ))*K), p.Results.m);
M = real(M);

end

function K = rbf_kernel(X,Z,varargin)

p = inputParser;
addParameter(p, 'theta', 1);
parse(p, varargin{:});

Kst = exp(-pdist2(X', Z')/(2*p.Results.theta.^2));
K = [exp(-pdist2(X', X')/(2*p.Results.theta.^2)) Kst; Kst' exp(-pdist2(Z', Z')/(2*p.Results.theta.^2))];


end


