function [beta] = kmm_est(XQ, XP, theta, kernel)
% Use Kernel Mean Matching to estimate weights for importance weighting.
% Jiayuan Huang, Alex Smola, Arthur Gretton, Karsten Borgwardt & Bernhard
% Schoelkopf. Correcting Sample Selection Bias by unlabeled data. 

[~,NQ] = size(XQ);
[~,NP] = size(XP);

K = zeros(NQ,NQ);
k = zeros(NQ,NP);
switch kernel
    case 'rbf'
        for i = 1:NQ
            for j = 1:NQ
                K(i,j) = exp(-sum((XQ(:,i)-XQ(:,j)).^2)./(2*theta));
            end
            for j = 1:NP
                k(i,j) = exp(-sum((XQ(:,i)-XP(:,j)).^2)./(2*theta)); 
            end
        end
end
k = NQ./NP*sum(K,2);

% Solve quadratic program
beta = quadprog(K,k,ones(1,NQ),NQ./sqrt(NQ),[],[],zeros(NQ,1),ones(NQ,1))';

end