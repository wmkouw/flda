function [W,TX,TZ,pred,varargout] = tca_svm(X,Z,yX,varargin)
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


addpath(genpath('../tcaPackage'));

% Parse input
p = inputParser;
addParameter(p, 'yZ', []);
addParameter(p, 'V', []);
addParameter(p, 'yV', []);
addParameter(p, 'nE', []);
addParameter(p, 'Kt', 'rbf');
addParameter(p, 'kP', 1);
addParameter(p, 'mu', 1);
addParameter(p, 'l2', 0);
parse(p, varargin{:});

% Cast to full
X = full(X);
Z = full(Z);

% Shapes
[~,NX] = size(X);
[~,NZ] = size(Z);

% Subspace disagreement measure
if isempty(p.Results.nE)
    
    % Principal Components
    [PX,~] = pca(X', 'Economy', false);
    [PZ,~] = pca(Z', 'Economy', false);
    [PP,~] = pca([X'; Z'], 'Economy', false);
    
    % Find subspace disagreement
    alpha = acos(diag(PX'*PP));
    beta = acos(diag(PZ'*PP));
    SDM = 0.5*(sin(alpha) + sin(beta));
    dix = find(SDM>0.999,1,'first');
    if isempty(dix); dix = length(SDM); end
    disp(['Optimal dimensionality (SDM): ' num2str(dix)]);
else
    dix = p.Results.nE;
end

% Set TCA options
fprintf('TCA based Feature Extraction \n');
options = tca_options('Kernel', p.Results.Kt, 'KernelParam', p.Results.kP, 'Mu', p.Results.mu, 'lambda', 0, 'Dim', dix);

% Find components
if ~isempty(p.Results.V)
    [TX,TZ,TV] = tca(X', Z', options, p.Results.V');
    varargout{2} = TV;
else
    [TX,~,TZ] = tca(X', Z', options, Z');
end
fprintf('Found Transfer Components \n');

% Apply Maximum Smoothing Principle for bandwidth selection
% gamma = median(pdist(TX).^2);

% Crossvalidate over gamma
gamma = [0.1 1 10 100];
lG = length(gamma);
valacc = zeros(1,lG);
for g = 1:lG
   valacc(g) = svmtrain(yX, TX, ['-c ' num2str(p.Results.l2) ' -t 2 -q -v 2 -g ' num2str(gamma(g))]);
end
[~,gix] = max(valacc);
disp(['Optimal gamma ' num2str(gamma(gix))]);

% Train gamma optimized svm
W = svmtrain(yX,TX, ['-c ' num2str(p.Results.l2) ' -t 2 -q -g ' num2str(gamma(gix))]);

% Do classification on target set
if ~isempty(p.Results.yZ);
    [pred,acc,~] = svmpredict(p.Results.yZ, TZ, W);
    varargout{1} = (100-acc(1))./100;
else
    [pred] = svmpredict(zeros(NZ,1), TZ, W, []);
end

if ~isempty(p.Results.V)
    [pred,acc,~] = svmpredict(p.Results.yV, TV, W);
    varargout{3} = pred';
else
    
    
end
