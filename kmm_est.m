function [beta] = kmm_est(XQ, XP, theta, kernel)
% Use Kernel Mean Matching to estimate weights for importance weighting.
% Jiayuan Huang, Alex Smola, Arthur Gretton, Karsten Borgwardt & Bernhard
% Schoelkopf. Correcting Sample Selection Bias by unlabeled data.
%
% The kernel computations come from PRTools

[~,NQ] = size(XQ);
[~,NP] = size(XP);

switch kernel
    case 'rbf'
        
        % Calculate Euclidean distances
        K = diste(XQ);
        k = diste(XQ,XP);
        
        % Cleanup
        I = find(K<0); K(I) = zeros(size(I));
        J = find(K<0); K(J) = zeros(size(J));
        
        % Radial basis function
        K = exp(-K/(2*theta.^2));
        k = exp(-k/(2*theta.^2));
        k = NQ./NP*sum(k,2);
        
    case 'diste'
        % Calculate Euclidean distances
        K = diste(XQ);
        k = diste(XQ,XP);
        if theta ~= 2
            K = sqrt(K).^theta;
            k = sqrt(k).^theta;
        end
        k = NQ./NP*sum(k,2);
end

% Approximate if memory shortage
a = whos('K');
if a.bytes > 2e9;
    K(K<.2) = 0;
    K = sparse(K);
end

% Solve quadratic program
options.Display = 'iter';
beta = quadprog(K,k,[ones(1,NQ); -ones(1,NQ)],[NQ./sqrt(NQ)+NQ NQ./sqrt(NQ)-NQ],[],[],zeros(NQ,1),ones(NQ,1), [], options)';

end


function D = diste(X1,varargin)
% Calculates Euclidean distance between vectors

addpath(genpath('pdistc'));
addpath(genpath('~/Codes/pdistc'));

if nargin==1
    D = squareform(pdistc(full(X1)));
else
    D = pdist2(X1',varargin{1}');
end

end