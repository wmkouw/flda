function [W,theta] = flda_qd_blankout(XQ,XP,y,l2)
% Function to calculate feature importance regularized least squares classifier
% Note: no optimization is necessary for ls

% Shape
[D,N] = size(XQ);

% Number of classes
K = max(y);

% Check for vector y
if size(y,1)~=N; y = y'; end

if K==2
    
    % Check for y in {-1,+1}
    if ~isempty(setdiff(unique(y), [-1,1]));
        y(y~=1) = -1;
    end
    
    % Estimate dropout transfer parameters
    theta = theta_b(XQ,XP);

    % Second moment of transfer distribution
    VX = diag(theta./(1-theta)).*(XQ*XQ');

    % Least squares solution
    if N<D
        XI = pinv(XQ*XQ' + VX + l2*eye(size(XQ,1)));
    else
        XI = inv(XQ*XQ' + VX + l2*eye(size(XQ,1)));
    end 
    W = XI*XQ*y;
    W = [W -W];
else
    
    W = zeros(D,K);
    theta = cell(1,K);
    for k = 1:K
        
        % Labels
        yk = (y==k);
        
        % 50up-50down resampling
        ix = randsample(find(yk==0), floor(.5*sum(1-yk)));
        Xk = [XQ(:,ix) repmat(XQ(:,yk), [1 floor((K-1)/2)])];
        yk = [double(yk(ix)); ones(1,floor((K-1)./2)*sum(yk))'];
        yk(yk==0) = -1;
        
        % Estimate dropout transfer parameters
        theta{k} = theta_b(Xk,XP);
        
        % Second moment of transfer distribution
        VX = diag(theta{k}./(1-theta{k})).*(Xk*Xk');

        % Least squares solution
        if N<D
            XI = pinv(Xk*Xk' + VX + l2*eye(size(Xk,1)));
        else
            XI = inv(Xk*Xk' + VX + l2*eye(size(Xk,1)));
        end        
        W(:,k) = XI*Xk*yk;
        
    end

end

end

function [theta] = theta_b(XQ,XP)
% Function to estimate the parameters of a blankout transfer distribution

dumQ = XQ > 1e-10 | XQ < -1e-10;
dumP = XP > 1e-10 | XP < -1e-10;

theta = min(0.9999,max(0,1-mean(dumP,2)./mean(dumQ,2)));

end
