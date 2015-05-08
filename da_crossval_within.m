function [W,Theta,tcl_err] = da_crossval_within(clf,XQ,yQ,varargin)
% Function to do crossvalidation using a domain adaptive classifier
%
% Assumes bias has already been added

% Parse hyperparameters
p = inputParser;
addOptional(p, 'nR', 5);
addOptional(p, 'nF', 5);
addOptional(p, 'l2', 1e-3);
addOptional(p, 'nE', 100);
addOptional(p, 'h1', 30);
addOptional(p, 'h2', 25);
addOptional(p, 'si', 1);
parse(p, varargin{:});

% Really don't like long variable names...
l2 = p.Results.l2;
nR = p.Results.nR;
nF = p.Results.nF;
nE = p.Results.nE;
h1 = p.Results.h1;
h2 = p.Results.h2;
si = p.Results.si;

% Shape
[~,NQ] = size(XQ);

% Preallocation
W = cell(nR,nF);
Theta = cell(nR,nF);
tcl_err = inf*ones(nR,NQ);
for r = 1:nR
    
    % Permute and create folds
    folds = ceil(randperm(NQ)./ (NQ./nF));
    
    for f = 1:nF
        
        % Update progress
        disp(['Fold ' num2str(f) '/' num2str(nF) ' of repeat ' num2str(r) '/' num2str(nR)]);
        
        % Split off validation set
        out_X = XQ(:,folds==f);
        out_y = yQ(folds==f);
        
        % Split off training set
        if nF==1
            in_X = out_X;
            in_y = out_y;
        else
            in_X = XQ(:,folds~=f);
            in_y = yQ(folds~=f);
        end
        
        switch clf
            case 'qd'
                W{r,f} = qd_1vall(in_X,in_y,l2);
                Theta{r,f} = [];
            case 'qd_d'
                [W{r,f}, Theta{r,f}] = fir_qd_dropout(in_X,in_X,in_y,l2);
            case 'qd_b'
                [W{r,f}, Theta{r,f}] = fir_qd_blankout(in_X,in_X,in_y,l2);
            case 'log'
                W{r,f} = mLogR(in_X,in_y,l2);
                Theta{r,f} = [];
            case 'log_d'
                [W{r,f},Theta{r,f}] = fir_log_dropout(in_X,in_X,in_y',l2);
            case 'log_b'
                [W{r,f},Theta{r,f}] = fir_log_blankout(in_X,in_X,in_y',l2);
            case 'scl'
                [W{r,f},Theta{r,f},~] = scl(in_X(1:end-1,:),in_X(1:end-1,:),in_y,l2,h1,h2);
                out_X = [out_X(1:end-1,:); Theta{r,f}*out_X(1:end-1,:); ones(1,size(out_X,2))];
            case 'kmm'
                Theta{r,f} = kmm_est(in_X,out_X,si, 'rbf');
                W{r,f} = irw_log(in_X,in_y,Theta{r,f},l2);
            case 'irw'
                Theta{r,f} = irw_est(in_X,in_X);
                W{r,f} = irw_log(in_X,in_y,Theta{r,f},l2);
            case 'sa'
                [W{r,f},EQ,EP] = subspace_alignment(in_X,in_X,in_y,l2,nE);
                Theta{r,f} = {EQ,EP};
                out_X = [(out_X'*EP)'; ones(1,size(out_X,2))];
            otherwise
                error(['Classifier ' clf ' not implemented']);
        end
        
        % Classification error on held out set
        [~,pred] = max(W{r,f}'*out_X, [], 1);
        tcl_err(r,folds==f) = (pred ~= out_y');
        
    end
end

end