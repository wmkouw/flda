function [W] = qd_1vall(X,y,l2)
% Note: no optimization is necessary for ls

% Shape
[D,N] = size(X);

% Number of classes
K = numel(unique(y));

if K==2
    
    % Check for y in {-1,+1}
    if ~isempty(setdiff(unique(y), [-1,1]));
        y(y~=1) = -1;
    end
    
    % Least squares solution
    if N<D
        XI = pinv(X*X' + N*l2*eye(size(X,1)));
    else
        XI = inv(X*X' + N*l2*eye(size(X,1)));
    end
    w = XI*X*y;
    W = [w -w];
else
    
    W = zeros(D,K);
    for k = 1:K
        
        % Labels
        yk = (y==k);
        
        % 50up-50down resampling
        ix = randsample(find(yk==0), floor(.5*sum(yk==0)));
        Xk = [X(:,ix) repmat(X(:,yk), [1 floor((K-1)/2)])];
        yk = [double(yk(ix)); ones(1,floor((K-1)./2)*sum(yk))'];
        yk(yk==0) = -1;
        
        % Least squares solution
        if N<D
            XI = pinv((Xk*Xk' + l2*eye(size(Xk,1))));
        else
            XI = inv((Xk*Xk' + l2*eye(size(Xk,1))));
        end
        W(:,k) = XI*Xk*yk;
        
    end

end

end
