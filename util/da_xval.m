function [W,Theta,err,mis,l2_opt] = da_xval(clf,X,yX,Z,yZ,varargin)
% Function to do crossvalidation using a domain adaptive classifier
% Assumes MxN data

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

% Really don't like long variable names...
nR = p.Results.nR;
nF = p.Results.nF;
nE = p.Results.nE;
m = p.Results.m;
h = p.Results.h;
si = p.Results.si;
l2 = p.Results.l2;
nL = length(l2);

% Add bias if necessary
if ~all(X(end,:)==1); X = [X; ones(1,size(X,2))]; end
if ~all(Z(end,:)==1); Z = [Z; ones(1,size(Z,2))]; end

% Shape
[~,NX] = size(X);
[~,NZ] = size(Z);

% Preallocation
W = cell(nR,nF,nL);
Theta = cell(nR,nF,nL);
mis = NaN(nR,NX,nL);

% Loop through l2 regularization lambda values
if ~((nL==1) && (nF==1))
    for l = 1:nL
        % Repeat
        for r = 1:nR
            
            % Permute and create folds
            folds = ceil(randperm(NX)./ (NX./nF));
            
            % Loop through folds
            for f = 1:nF
                
                % Update progress
                disp(['Fold ' num2str(f) '/' num2str(nF) ' of repeat ' num2str(r) '/' num2str(nR)]);
                
                % Split off validation set
                val_X = X(:,folds==f);
                val_y = yX(folds==f);
                
                % Split off training set
                if nF==1
                    trn_X = val_X;
                    trn_y = val_y;
                else
                    trn_X = X(:,folds~=f);
                    trn_y = yX(folds~=f);
                end
                
                switch clf
                    case {'qd','tqd'}
                        % Train 1 vs. all Fisher classifier
                        W{r,f,l} = qd_1vall(trn_X,trn_y, 'l2', l2(l));
                        Theta{r,f,l} = [];
                        
                        % Make predictions on validation set
                        [~,pred] = max(W{r,f,l}'*val_X, [], 1);
                        
                    case {'lr','tlr'}
                        % Train a multiclass logistic regressor
                        W{r,f,l} = mLR(trn_X',trn_y, 'l2', l2(l));
                        Theta{r,f,l} = [];
                        
                        % Classification error on validation set
                        [~,pred] = max(W{r,f,l}'*val_X, [], 1);
                    case 'svm'
                        % Crossvalidate over kernel bandwidth param
                        if l2(l)==0; l2(l) = realmin; end
                        lG = length(p.Results.Ga);
                        valacc = zeros(1,lG);
                        for g = 1:lG
                            valacc(g) = svmtrain(trn_y, trn_X', ['-c ' num2str(l2(l)) ' -v 2 -q -g ' num2str(p.Results.Ga(g))]);
                        end
                        [~,ixGa] = max(valacc);
                        Theta{r,f,l} = [];
                        
                        % Train model
                        W{r,f,l} = svmtrain(trn_y,trn_X',['-c ' num2str(l2(l)) ' -g ' num2str(p.Results.Ga(ixGa))]);
                        pred = svmpredict(val_y,val_X',W{r,f,l})';
                        
                    case 'flda_log_d'
                        % Train fir logistic with dropout
                        [W{r,f,l},Theta{r,f,l}] = flda_log_dropout(trn_X,Z,trn_y',l2(l));
                        
                        % Classification error on validation set
                        [~,pred] = max(W{r,f,l}'*val_X, [], 1);
                        
                    case 'flda_log_b'
                        % Train fir logistic with blankout
                        [W{r,f,l},Theta{r,f,l}] = flda_log_blankout(trn_X,Z,trn_y',l2(l));
                        
                        % Classification error on validation set
                        [~,pred] = max(W{r,f,l}'*val_X, [], 1);
                    case 'mFLDA_log_b'
                        % Train fir logistic with blankout
                        [W{r,f,l},Theta{r,f,l}] = mFLDA_log_b(trn_X',Z',trn_y', 'l2', l2(l));
                        
                        % Classification error on validation set
                        [~,pred] = max(W{r,f,l}'*val_X, [], 1);
                    case 'flda_qd_d'
                        % Train fir quadratic with dropout
                        [W{r,f,l}, Theta{r,f,l}] = flda_qd_dropout(trn_X,Z,trn_y',l2(l));
                        
                        % Classification error on validation set
                        [~,pred] = max(W{r,f,l}'*val_X, [], 1);
                        
                    case 'flda_qd_b'
                        % Train fir quadratic with blankout
                        [W{r,f,l}, Theta{r,f,l}] = flda_qd_blankout(trn_X,Z,trn_y',l2(l));
                        
                        % Classification error on validation set
                        [~,pred] = max(W{r,f,l}'*val_X, [], 1);
                        
                    case 'fir_qd_mimp'
                        % Train fir with mean imputation
                        [W{r,f,l}, Theta{r,f,l}] = flda_qd_mimp(trn_X,Z,trn_y',l2(l),p.Results.mu);
                        
                        % Classification error on validation set
                        [~,pred] = max(W{r,f,l}'*val_X, [], 1);
                        
                    case 'scl'
                        % Train a structure correspondence learner
                        [W{r,f,l},Theta{r,f,l},~] = scl(trn_X(1:end-1,:),Z(1:end-1,:),trn_y,l2(l), 'm', p.Results.m, 'h', p.Results.h);
                        val_X = [val_X(1:end-1,:); Theta{r,f,l}*val_X(1:end-1,:); ones(1,size(val_X,2))];
                        
                        % Classification error on validation set
                        [~,pred] = max(W{r,f,l}'*val_X, [], 1);
                        
                    case 'kmm'
                        % Train an instance reweighted lr with kmm weights
                        [Theta{r,f,l}] = irw_est_kmm(trn_X,Z, 'theta', si, 'kernel', 'rbf');
                        W{r,f,l} = mWLR(trn_X(1:end-1,:)',trn_y, 'iw', Theta{r,f,l},'l2', l2(l));
                        
                        % Classification error on validation set
                        [~,pred] = max(W{r,f,l}'*val_X, [], 1);
                        
                    case 'irw_log'
                        % Train an instance reweighted lr with log weights
                        Theta{r,f,l} = irw_est_log(trn_X,Z, .1);
                        W{r,f,l} = mWLR(trn_X(1:end-1,:),trn_y,Theta{r,f,l},'l2', l2(l));
                        
                        % Classification error on validation set
                        [~,pred] = max(W{r,f,l}'*val_X, [], 1);
                        
                    case 'sa_svm'
                        % Train a subspace aligned classifier
                        if l2(l)==0; l2(l) = realmin; end
                        [W,Theta,~,~] = sa_svm(X',Z',yX, 'yZ', yZ, 'l2', l2(l), 'nE', nE);
                        [pred] = svmpredict(val_y,val_X'*Theta*val_X,W)';
                    case 'tca_lr'
                        % Transfer component analysis with logistic loss
                        [W{r,f,l},~,~,~,~,~,pred] = tca_lr(trn_X(1:end-1,:), Z(1:end-1,:), trn_y, 'l2', l2(l), 'V', val_X(1:end-1,:), 'dim', p.Results.nE);
                        
                    case 'tca_svm'
                        % Transfer component analysis with svm
                        if l2(l)==0; l2(l) = realmin; end
                        [W{r,f,l},~,~,~,~,~,pred] = tca_svm_hyp(trn_X(1:end-1,:), Z(1:end-1,:), trn_y, 'l2', l2(l), 'V', val_X(1:end-1,:), 'yV', val_y, 'nE', p.Results.nE, 'mu', p.Results.mu, 'La', p.Results.La, 'Ga', p.Results.Ga, 'Kp', p.Results.Kp, 'Kt', p.Results.Kt, 'rndhyp', 0);
                        
                    case {'gfk_knn','gfk_lr'}
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
l2_opt = l2(ixL2);

% Train on full source set using optimal
err = zeros(1,nR);
for r = 1:nR
    switch clf
        case 'tqd'
            % Target quadratic's error is the optimal crossvalidated error
            err(r) = mean(mean(mis(:,:,ixL2),2),1);
        case 'tlr'
            % Target logistic's error is the optimal crossvalidated error
            err(r) = mean(mean(mis(:,:,ixL2),2),1);
        case 'qd'
            % 1 vs all Fisher classifier
            W = qd_1vall(X,yX,l2(ixL2));
            Theta = [];
            [~,pred] = max(W'*Z, [], 1);
            err(r) = mean(pred ~= yZ');
        case 'knn'
            % 1-Nearest neighbour
            D = pdist2(X',Z');
            [~,ixL2] = min(D);
            pred = yX(ixL2);
            err(r) = mean(pred ~= yZ);
        case 'svm'
            % Train a radial basis support vector machine
            if l2(ixL2)==0; l2(ixL2) = realmin; end
            valacc = zeros(1,lG);
            for g = 1:lG
                valacc(g) = svmtrain(trn_y, trn_X', ['-c ' num2str(l2(ixL2)) ' -v 2 -q -g ' num2str(p.Results.Ga(g))]);
            end
            [~,ixGa] = max(valacc);
            W = svmtrain(yX, X', ['-c ' num2str(l2(ixL2)) ' -q -g ' num2str(p.Results.Ga(ixGa))]);
            Theta = [];
            [~,acc,~] = svmpredict(yZ,Z',W);
            err(r) = (100-acc(1))./100;
        case 'lr'
            % Multiclass logistic regressor
            W = mLR(X',yX,'l2',l2(ixL2));
            Theta = [];
            [~,pred] = max(Z'*W, [], 2);
            err(r) = mean(pred ~= yZ);
        case 'irw_log'
            % Instance reweighted lr with logistic discrimination weights
            Theta = irw_est_log(X,Z,.1);
            W = mWLR(X',yX, 'iw', Theta,'l2', l2(ixL2));
            [~,pred] = max(W'*Z, [], 1);
            err(r) = mean(pred ~= yZ');
        case 'kmm'
            % Instance reweighted lr with kernel mean matched weights
            [Theta] = irw_est_kmm(X,Z, 'theta', si, 'kernel', 'rbf');
            W = mWLR(X',yX, 'iw', Theta,'l2', l2(ixL2));
            [~,pred] = max(W'*Z, [], 1);
            err(r) = mean(pred ~= yZ');
        case 'scl'
            % Structural Correspondence Learning with Logistic Regression
            [W,Theta,~] = scl(X(1:end-1,:),Z(1:end-1,:),yX, 'l2', l2(ixL2), 'm', m, 'h' ,h);
            [~,pred] = max(W'*[Z(1:end-1,:); Theta*Z(1:end-1,:); ones(1,NZ)], [], 1);
            err(r) = mean(pred ~= yZ');
        case 'sa_svm'
            % Subspace aligned lr
            if l2(ixL2)==0; l2(ixL2) = realmin; end
            [W,Theta,pred,err(r)] = sa_svm(X(1:end-1,:)',Z(1:end-1,:)',yX, 'yZ', yZ, 'l2', l2(ixL2), 'nE', nE);
        case 'gfk_knn'
            % Geodesic flow kernel with kernel 1-nn
            [pred,Theta] = gfk(X(1:end-1,:), Z(1:end-1,:), yX, 'clf', '1-nn', 'nE', nE);
            W = [];
            err(r) = mean(pred ~= yZ');
        case 'gfk_lr'
            % Geodesic flow kernel with logistic regression
            [pred,Theta] = gfk(X(1:end-1,:), Z(1:end-1,:), yX, 'clf', 'log');
            W = [];
            err(r) = mean(pred ~= yZ);
        case 'gfk_svm'
            % Geodesic flow kernel with logistic regression
            [pred,Theta] = gfk(X(1:end-1,:), Z(1:end-1,:), yX, 'clf', 'svm');
            W = [];
            err(r) = mean(pred ~= yZ);
        case 'tca_lr'
            % Transfer component analysis classifier
            [W,~,~,~,err(r)] = tca_lr(X(1:end-1,:), Z(1:end-1,:), yX, 'yZ', yZ, 'l2', l2(ixL2), 'dim', nE);
        case 'tca_svm'
            % Transfer component analysis classifier
            if l2(ixL2)==0; l2(ixL2) = realmin; end
            [W,~,~,~,err(r)] = tca_svm_hyp(X(1:end-1,:), Z(1:end-1,:), yX, 'yZ', yZ, 'l2', l2(ixL2), 'nE', nE, 'mu', p.Results.mu, 'La', p.Results.La, 'Ga', p.Results.Ga, 'Kp', p.Results.Kp, 'Kt', p.Results.Kt, 'rndhyp', 0);
        case 'flda_qd_d'
            % fir quadratic with dropout
            [W, Theta] = flda_qd_dropout(X,Z,yX, l2(ixL2));
            [~,pred] = max(W'*Z, [], 1);
            err(r) = mean(pred ~= yZ');
        case 'flda_qd_b'
            % fir quadratic with blankout
            [W,Theta] = flda_qd_blankout(X,Z,yX', l2(ixL2));
            [~,pred] = max(W'*Z, [], 1);
            err(r) = mean(pred ~= yZ');
        case 'flda_qd_mimp'
            % Train fir with mean imputation
            [W, Theta] = flda_qd_mimp(X,Z,yX',l2(ixL2),p.Results.mu);
            [~,pred] = max(W'*Z, [], 1);
            err(r) = mean(pred ~= yZ');
        case 'flda_log_d'
            % fir logistic with dropout
            [W,Theta] = flda_log_dropout(X,Z,yX', l2(ixL2));
            [~,pred] = max(W'*Z, [], 1);
            err(r) = mean(pred ~= yZ');
        case 'flda_log_b'
            % fir logistic with blankout
            [W,Theta] = flda_log_blankout(X,Z,yX',l2(ixL2));
            [~,pred] = max(W'*Z, [], 1);
            err(r) = mean(pred ~= yZ');
        case 'mFLDA_log_b'
            % fir logistic with blankout
            [W,Theta] = mFLDA_log_b(X',Z',yX', 'l2', l2(ixL2));
            [~,pred] = max(Z'*W, [], 2);
            err(r) = mean(pred ~= yZ);
        otherwise
            error(['Classifier not found']);
    end
end

err = mean(err);

end
