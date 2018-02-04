function [W,Theta,err,mis,l2_opt] = da_xval(clf,X,yX,Z,yZ,varargin)
% Crossvalidation over domains

% Parse hyperparameters
p = inputParser;
addOptional(p, 'nR', 1);
addOptional(p, 'nF', 1);
addOptional(p, 'l2', 1);
addOptional(p, 'nE', 100);
addOptional(p, 'm', 20);
addOptional(p, 'h', 15);
addOptional(p, 'si', 1);
addOptional(p, 'mu', 1);
addOptional(p, 'La', 1);
addOptional(p, 'Ga', 1);
addOptional(p, 'Kt', 'rbf');
addOptional(p, 'Kp', 1);
parse(p, varargin{:});

% Number of regularization parameters
nL = length(p.Results.l2);

% Add bias if necessary
if ~all(X(end,:)==1); X = [X; ones(1,size(X,2))]; end
if ~all(Z(end,:)==1); Z = [Z; ones(1,size(Z,2))]; end

% Shape
[~,NX] = size(X);
[~,NZ] = size(Z);

% Preallocation
W = cell(p.Results.nR, p.Results.nF, nL);
Theta = cell(p.Results.nR, p.Results.nF, nL);
mis = NaN(p.Results.nR, NX, nL);

% Loop through l2 regularization lambda values
if ~((nL==1) && (p.Results.nF==1))
    for l = 1:nL
        disp(['lambda = ' num2str(p.Results.l2(l))]);
        
        % Repeat
        for r = 1:p.Results.nR
            
            % Permute and create folds
            folds = ceil(randperm(NX)./ (NX./p.Results.nF));
            
            % Loop through folds
            for f = 1:p.Results.nF
                
                % Update progress
                disp(['Fold ' num2str(f) '/' num2str(p.Results.nF) ' of repeat ' ...
                    num2str(r) '/' num2str(p.Results.nR)]);
                
                % Split off validation set
                val_X = X(:,folds==f);
                val_y = yX(folds==f);
                
                % Split off training set
                if p.Results.nF==1
                    trn_X = val_X;
                    trn_y = val_y;
                else
                    trn_X = X(:,folds~=f);
                    trn_y = yX(folds~=f);
                end
                
                switch clf
                    case {'qd','tqd'}
                        % Train 1 vs. all Fisher classifier
                        W{r,f,l} = qd_1vall(trn_X,trn_y, 'l2', p.Results.l2(l));
                        Theta{r,f,l} = [];
                        
                        % Make predictions on validation set
                        [~,pred] = max(W{r,f,l}'*val_X, [], 1);
                        
                    case {'lr','tlr'}
                        % Train a multiclass logistic regressor
                        W{r,f,l} = mLR(trn_X',trn_y, 'l2', p.Results.l2(l));
                        Theta{r,f,l} = [];
                        
                        % Classification error on validation set
                        [~,pred] = max(W{r,f,l}'*val_X, [], 1);
                        
                    case 'flda_log_d'
                        % Train flda logistic with dropout
                        [W{r,f,l},Theta{r,f,l}] = flda_log_dropout(trn_X,Z,trn_y', p.Results.l2(l));
                        
                        % Classification error on validation set
                        [~,pred] = max(W{r,f,l}'*val_X, [], 1);
                        
                    case 'flda_log_b'
                        % Train flda logistic with blankout
                        [W{r,f,l},Theta{r,f,l}] = flda_log_blankout(trn_X,Z,trn_y', p.Results.l2(l));
                        
                        % Classification error on validation set
                        [~,pred] = max(W{r,f,l}'*val_X, [], 1);
                        
                    case 'flda_qd_d'
                        % Train flda quadratic with dropout
                        [W{r,f,l}, Theta{r,f,l}] = flda_qd_dropout(trn_X,Z,trn_y', p.Results.l2(l));
                        
                        % Classification error on validation set
                        [~,pred] = max(W{r,f,l}'*val_X, [], 1);
                        
                    case 'flda_qd_b'
                        % Train flda quadratic with blankout
                        [W{r,f,l}, Theta{r,f,l}] = flda_qd_blankout(trn_X,Z,trn_y', p.Results.l2(l));
                        
                        % Classification error on validation set
                        [~,pred] = max(W{r,f,l}'*val_X, [], 1);
                        
                    case 'scl'
                        % Train a structure correspondence learner
                        [W{r,f,l},Theta{r,f,l},~] = scl(trn_X(1:end-1,:),Z(1:end-1,:),trn_y, p.Results.l2(l), ...
                            'm', p.Results.m, 'h', p.Results.h);
                        val_X = [val_X(1:end-1,:); Theta{r,f,l}*val_X(1:end-1,:); ones(1,size(val_X,2))];
                        
                        % Classification error on validation set
                        [~,pred] = max(W{r,f,l}'*val_X, [], 1);
                        
                    case 'kmm'
                        % Train an instance reweighted lr with kmm weights
                        [Theta{r,f,l}] = irw_est_kmm(trn_X,Z, 'theta', p.Results.si, 'kernel', 'rbf');
                        W{r,f,l} = mWLR(trn_X(1:end-1,:)',trn_y, 'iw', Theta{r,f,l},'l2', p.Results.l2(l));
                        
                        % Classification error on validation set
                        [~,pred] = max(W{r,f,l}'*val_X, [], 1);
                        
                    case 'sa_svm'
                        % Train a subspace aligned classifier
                        [W,Theta,~,~] = sa_svm(X',Z',yX, 'yZ', yZ, 'l2', p.Results.l2(l), 'nE', p.Results.nE);
                        [pred] = svmpredict(val_y,val_X'*Theta*val_X,W)';
                        
                    case 'tca_svm'
                        % Transfer component analysis with svm
                        [W{r,f,l},~,~,~,~,~,pred] = tca_svm(trn_X(1:end-1,:), Z(1:end-1,:), trn_y, ...
                            'l2', p.Results.l2(l), 'V', val_X(1:end-1,:), 'yV', val_y, 'nE', p.Results.nE, ...
                            'mu', p.Results.mu, 'Kp', p.Results.Kp, 'Kt', p.Results.Kt);
                        
                    case 'gfk_knn'
                        pred = NaN(size(val_y'));
                        
                    otherwise
                        disp(['No crossvalidation']);
                end
                
                % Check predictions
                mis(r,folds==f,l) = (pred ~= val_y');
            end
        end
    end
end

% Select optimal regularization parameter
[~,ixL2] = min(mean(mean(mis,2), 1));
l2_opt = p.Results.l2(ixL2);

% Train on full source set using optimal
err = zeros(1,p.Results.nR);
for r = 1:p.Results.nR
    switch clf
        case 'tqd'
            % Target quadratic's error is the optimal crossvalidated error
            err(r) = mean(mean(mis(:,:,ixL2),2),1);
            
        case 'tlr'
            % Target logistic's error is the optimal crossvalidated error
            err(r) = mean(mean(mis(:,:,ixL2),2),1);
            
        case 'qd'
            % 1 vs all Fisher classifier
            W = qd_1vall(X,yX, l2_opt);
            Theta = [];
            [~,pred] = max(W'*Z, [], 1);
            err(r) = mean(pred ~= yZ');
            
        case 'lr'
            % Multiclass logistic regressor
            W = mLR(X',yX,'l2',l2_opt);
            Theta = [];
            [~,pred] = max(Z'*W, [], 2);
            err(r) = mean(pred ~= yZ);
            
        case 'kmm'
            % Instance reweighted lr with kernel mean matched weights
            [Theta] = irw_est_kmm(X,Z, 'theta', p.Results.si, 'kernel', 'rbf');
            W = mWLR(X',yX, 'iw', Theta,'l2', l2_opt);
            [~,pred] = max(W'*Z, [], 1);
            err(r) = mean(pred ~= yZ');
            
        case 'scl'
            % Structural Correspondence Learning with Logistic Regression
            [W,Theta,~] = scl(X(1:end-1,:),Z(1:end-1,:),yX, 'l2', l2_opt, ...
                'm', p.Results.m, 'h' , p.Results.h);
            [~,pred] = max(W'*[Z(1:end-1,:); Theta*Z(1:end-1,:); ones(1,NZ)], [], 1);
            err(r) = mean(pred ~= yZ');
            
        case 'sa_svm'
            % Subspace aligned lr
            [W,Theta, pred, err(r)] = sa_svm(X(1:end-1,:)',Z(1:end-1,:)',yX, 'yZ', yZ, ...
                'l2', l2_opt, 'nE', p.Results.nE);
            
        case 'gfk_knn'
            % Geodesic flow kernel with kernel 1-nn
            [pred,Theta] = gfk(X(1:end-1,:), Z(1:end-1,:), yX, 'clf', '1-nn', 'nE', p.Results.nE);
            W = [];
            err(r) = mean(pred ~= yZ');
            
        case 'tca_svm'
            % Transfer component analysis classifier
            if l2(ixL2)==0; l2(ixL2) = realmin; end
            [W,~,~,~,err(r)] = tca_svm(X(1:end-1,:), Z(1:end-1,:), yX, 'yZ', yZ, 'l2', l2(ixL2), ...
                'nE', nE, 'mu', p.Results.mu, 'La', p.Results.La, 'Ga', p.Results.Ga, 'Kp', p.Results.Kp, ...
                'Kt', p.Results.Kt);    
            
        case 'flda_qd_d'
            % flda quadratic with dropout
            [W, Theta] = flda_qd_dropout(X,Z,yX, l2_opt);
            [~,pred] = max(W'*Z, [], 1);
            err(r) = mean(pred ~= yZ');
            
        case 'flda_qd_b'
            % flda quadratic with blankout
            [W,Theta] = flda_qd_blankout(X,Z,yX', l2_opt);
            [~,pred] = max(W'*Z, [], 1);
            err(r) = mean(pred ~= yZ');
            
        case 'flda_log_d'
            % flda logistic with dropout
            [W,Theta] = flda_log_dropout(X,Z,yX', l2_opt);
            [~,pred] = max(W'*Z, [], 1);
            err(r) = mean(pred ~= yZ');
            
        case 'flda_log_b'
            % flda logistic with blankout
            [W,Theta] = flda_log_blankout(X,Z,yX',l2_opt);
            [~,pred] = max(W'*Z, [], 1);
            err(r) = mean(pred ~= yZ');
            
        otherwise
            error(['Classifier not found']);
    end
end

err = mean(err);

end
