function [F,theta,Pp] = scl(XQ,XP,yQ,l2,m,h)
% Function to calculate structural correspondence learning
% J. Blitzer,R. McDonald & F. Pereira (2006). Domain adaptation with
% Structural Correspondence Learning. EMNLP

addpath(genpath('minFunc'));

% Optimization options
options.DerivativeCheck = 'off';
options.Method = 'lbfgs';

% Shape
[M,~] = size(XQ);

% Number of classes
K = numel(unique(yQ));

% Check for y in {1,..K}
yQ(yQ== 0) = 2;
yQ(yQ==-1) = 2;

% Choose m pivot features;
[~,ix1] = sort(sum(XQ,2)+sum(XP,2), 'descend');
pivot = [XQ(ix1(1:m),:) XP(ix1(1:m),:)];
pivot(pivot>0) = 1;

% Solve m binary prediction tasks
Pp = zeros(M,m);
for l = 1:m
    disp(['Pivot feature #' num2str(l)]);
    Pp(:,l) = minFunc(@huber_grad, randn(M,1), options, [XQ XP], pivot(m,:), l2);
end

% Decompose pivot predictors
[U,~,~] = svd(Pp);
theta = U(:,1:h)';

% Minimize loss
f = minFunc(@mLogR_grad, randn((M+h+1)*K,1), options, [XQ; theta*XQ], yQ(:)', l2);

% Output MxK weight matrix
F = [reshape(f(1:end-K), [M+h K]); f(end-K+1:end)'];

end