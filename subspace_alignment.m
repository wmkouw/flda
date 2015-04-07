function [W,EQ,EP] = subspace_alignment(XQ,XP,yQ,l2,nE)
% Function to train a classifier on a subspace alignment matched domain
% set.
% Fernando, B., Habrard, A., Sebban, M. & Tuytelaars, T. Subspace alignment
% for domain adaption. ArXiv 2014

% Shapes
[MQ,NQ] = size(XQ);
[MP,NP] = size(XP);
K = numel(unique(yQ));

% Z-score domains
XQ = da_prep(XQ, 'zscore');
XP = da_prep(XP, 'zscore');

% Remove nans
XQ(isnan(XQ)) = 0;
XP(isnan(XP)) = 0;

% Covariance 
SQ = 1./NQ*(XQ*XQ'); 
SP = 1./NP*(XP*XP');

% Remove nans (result from sparsity)
SQ(isnan(SQ)) = 0;
SP(isnan(SP)) = 0;

% Eigenvectors
[EQ,~] = eigs(SQ,nE);
[EP,~] = eigs(SP,nE);

% Optimal linear transformation
M = EQ'*EP;

% Projecting source data on aligned space
XA = XQ'*EQ*M;

options.Method = 'lbfgs';
options.Display = 'valid';

% Classifier
yQ(yQ==-1) = 2;
W = minFunc(@mLR_grad, zeros((nE+1)*K,1), options, XA', yQ, l2);
if K ==2
    W = [W(1:nE); W(end-1)];
else
    W = [reshape(W(1:end-K), [nE K]); W(end-K+1:end)'];
end

end