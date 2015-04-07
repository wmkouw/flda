% Function to estimate reweighting coefficients for instance reweighting
function [iw] = irw_est(X0, X1)
% Function expects MxN matrices.
    
    % Shape
    [M0,N0] = size(X0);
    [M1,N1] = size(X1);
    Y = [ones(1,N0) 2*ones(1,N1)];
    
    % Run logistic regressor on domains
    options.DerivativeCheck = 'off';
    options.method = 'lbfgs';
    w = minFunc(@logreg_grad, randn((M0+1)*2,1), options, [X0 X1], Y, 0);
    w2 = [w(1:M0); w(end-1)];

    % Calculate posterior over samples of X0
    iw = 1./(1+exp(-w2'*[X0; ones(1,N0)]));
    
end

function [C, dC] = logreg_grad(x, train_X, train_labels, lambda, pos_E, pos_E_bias)
%LOGREG_GRAD Gradient of L2-regularized logistic regressor
%
%   [C, dC] = logreg_grad(x, train_X, train_labels, lambda, pos_E, pos_E_bias)
%
% Gradient of L2-regularized logistic regressor.


    % Decode solution
    [D, N] = size(train_X);
    K = numel(x) / (D + 1);
    E = reshape(x(1:D * K), [D K]);
    E_bias = reshape(x(D * K + 1:end), [1 K]);

    % Compute p(y|x)
    gamma = bsxfun(@plus, E' * train_X, E_bias');
    gamma = exp(bsxfun(@minus, gamma, max(gamma, [], 1)));
    gamma = bsxfun(@rdivide, gamma, max(sum(gamma, 1), realmin));
    
    % Compute conditional log-likelihood
    C = 0;
    for n=1:N
        C = C - log(max(gamma(train_labels(n), n), realmin));
    end
    C = C + lambda .* sum(x .^ 2);
    
    % Only compute gradient when required
    if nargout > 1
    
        % Compute positive part of gradient
        if ~exist('pos_E', 'var') || isempty(pos_E)
            pos_E = zeros(D, K);
            for k=1:K
                pos_E(:,k) = sum(train_X(:,train_labels == k), 2);
            end
        end
        if ~exist('pos_E_bias', 'var') || isempty(pos_E_bias)
            pos_E_bias = zeros(1, K);
            for k=1:K        
                pos_E_bias(k) = sum(train_labels == k);
            end
        end

        % Compute negative part of gradient    
        neg_E = train_X * gamma';
        neg_E_bias = sum(gamma, 2)';
        
        % Compute gradient
        dC = -[pos_E(:) - neg_E(:); pos_E_bias(:) - neg_E_bias(:)] + 2 .* lambda .* x;
    end    
end
