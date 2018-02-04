function [W,Theta,pred,err] = sa_svm(X,Z,yX,varargin)
% Function to train a domain adaptive classifier using Subspace Alignment
%
% Fernando, B., Habrard, A., Sebban, M. & Tuytelaars, T. Subspace alignment
% for domain adaption. ArXiv 2014
% Input:
%    X is source set in MxN format (no augmentation)
%    Z is target set in MxN format (no augmentation)
%    y is source label vector in [1,..K]
%    l2 is regularization parameter
%    nE is dimensionality to which the subspace is reduced to
% Output:
%    W is KxM resulting classifier
%    EX is source subspace
%    EZ is target subspace (must be used to map novel target data on)
%
% Wouter Kouw
% 15-09-2014

% Parse input
p = inputParser;
addParameter(p, 'yZ', []);
addParameter(p, 'V', []);
addParameter(p, 'yV', []);
addParameter(p, 'l2', 1e-3);
addParameter(p, 'nE', []);
parse(p, varargin{:});

% Shape
[N,D] = size(X);
[M,D] = size(Z);

% Prep
X = da_prep(X', {'sum_samp', 'zscore'})';
Z = da_prep(Z', {'sum_samp', 'zscore'})';    

% Principal Components
[PX,~] = pca(full(X), 'Economy', false);
[PZ,~] = pca(full(Z), 'Economy', false);
[PP,~] = pca(full([X; Z]), 'Economy', false);

% Subspace disagreement measure
if isempty(p.Results.nE);
    alpha = acos(diag(PX'*PP));
    beta = acos(diag(PZ'*PP));
    SDM = 0.5*(sin(alpha) + sin(beta));
    dix = find(SDM>0.999,1,'first');
    disp(['Optimal dimensionality (SDM): ' num2str(dix)]);
else
    dix = p.Results.nE;
end
PX = PX(:,1:dix);
PZ = PZ(:,1:dix);

% Similarity kernel
Theta = (PX*PX')*(PZ*PZ');

% Transform samples
SimX = X*Theta*X';
SimZ = X*Theta*Z';

% Augment
SimX = [[1:N]' SimX];
SimZ = [[1:M]' SimZ'];

% Train Support Vector Machine
W = svmtrain(yX, SimX, ['-t 4 -c ' num2str(p.Results.l2) ' -q']);

% Run predictions
if ~isempty(p.Results.yZ);
    [pred,acc,~] = svmpredict(p.Results.yZ, SimZ, W);
    err = (100-acc(1))./100;
else
    [pred] = svmpredict(zeros(), SimZ, W);
end

end


