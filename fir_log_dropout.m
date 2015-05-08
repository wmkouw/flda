function [W,theta] = fir_log_dropout(XQ,XP,yQ,lambda)
% Function to run the optimization procedure of the feature absence
% regularized domain adaptation classifier.

addpath(genpath('minFunc'));

% Optimization options
options.DerivativeCheck = 'off';
options.Method = 'lbfgs';
options.Display = 'valid';

% Shape
[MQ,~] = size(XQ);

% Number of classes
K = numel(unique(yQ));

if K==2
    
    % Estimate blankout transfer parameters
    theta = est_transfer_params_drop(XQ,XP);
    
    % Analytical solution to theta=1 => w=0
    ix = find(theta~=1);
    
    % Check for y in {-1,+1}
    if ~isempty(setdiff(unique(yQ), [-1,1]));
        yQ(yQ~=1) = -1;
    end
    
    % Minimize loss
    if ~isrow(yQ); yQ = yQ'; end
    options.DerivativeCheck = 'on';
    w = minFunc(@fir_log_dropout_grad, zeros(length(ix),1), options, XQ(ix,:),yQ,theta(ix),lambda);
    
    % Bookkeeping
    W = zeros(MQ,1);
    W(ix) = w;
    W = [W -W];
    
else
    
    W = zeros(MQ,K);
    for k = 1:K
        
        % Labels
        yk = (yQ==k);
        
        % 50up-50down resampling
        ix = randsample(find(yk==0), floor(.5*sum(1-yk)));
        Xk = [XQ(:,ix) repmat(XQ(:,yk), [1 floor((K-1)/2)])];
        yk = [double(yk(ix))'; ones(floor((K-1)./2)*sum(yk),1)];
        yk(yk==0) = -1;
        
        % Estimate blankout transfer parameters
        theta = est_transfer_params_drop(Xk,XP);
        
        % Analytical solution to theta=1 => w=0
        ix = find(theta~=1);
        
        % Minimize loss
        if ~isrow(yk); yk = yk'; end
        w = minFunc(@fir_log_dropout_grad, zeros(length(ix),1), options, Xk(ix,:),yk,theta(ix),lambda);
        
        % Bookkeeping
        W(ix,k) = w;
    end
end

end