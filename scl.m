function [F,theta,Pp] = scl(XQ,XP,yQ,l2,m,h)
% Function to calculate structural correspondence learning
% J. Blitzer,R. McDonald & F. Pereira (2006). Domain adaptation with
% Structural Correspondence Learning. EMNLP

% Concat data from both domains
X = [XQ XP];
[M,~] = size(X);

% Choose m pivot features;
[~,ix1] = sort(sum(X,2), 'descend');
pivot = [XQ(ix1(1:m),:) XP(ix1(1:m),:)];
pivot(pivot>0) = 1;

% Solve m binary prediction tasks
options.DerivativeCheck = 'off';
options.Method = 'lbfgs';
Pp = zeros(M,m);
for l = 1:m
    disp(['Pivot feature #' num2str(l)]);
    Pp(:,l) = minFunc(@huber_grad, randn(M,1), options, X, pivot(m,:), l2);
end

% Decompose pivot predictors
[U,~,~] = svd(Pp);
theta = U(:,1:h)';

% Solve larger problem
K = numel(unique(yQ));
if K==2
    f = minFunc(@mLR_grad, randn(M+h+1,1), options, [XQ; theta*XQ], yQ(:)', l2);
    F = [f(1:(M+h)); f(end-1)];
else
    f = minFunc(@mLR_grad, randn((M+h+1)*K,1), options, [XQ; theta*XQ], yQ(:)', l2);
    F = [reshape(f(1:end-K), [M+h K]); f(end-K+1:end)'];
end

end