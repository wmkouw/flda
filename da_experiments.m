function da_experiments(dataset, prep, learner, lambda, sv)
% Function to run domain adaptation experiments
% Wouter Kouw
% 04-10-2014

% Parse inputs
if ~exist('lambda', 'var');
    lambda = [0 .1 1];
end
if ~exist('sv', 'var');
    sv = '';
end

% Add path
if (isempty(which('minFunc')) || isempty(which('minConf_TMP')) || isempty(which('FARq_grad')))
    addpath(genpath('./code'));
    addpath(genpath('minFunc'));
    addpath(genpath('minConf'));
    addpath(genpath('libsvm-3.11'));
    addpath(genpath('/home/wmkouw/Data/'));
end

domains = {'books', 'dvd', 'electronics', 'kitchen'};
nC = 2;
L = length(domains);
L2 = length(lambda);
tst_err = zeros(L,L,L2);
trn_err = zeros(L,L,L2);
for i = 1:L
    for j = 2:L
        % Display progress
        disp(['Running ' domains{i} ' -> ' domains{j}]);
        disp('Loading and slicing domains...');
        switch lower(dataset)
            case 'amazon'
                
                try
                    load('amazon.mat');
                catch
                    load('~/Data/Amazon/amazon.mat');
                end
                yy(yy==-1) = 2;
                
                XQ = xx(:,offset(i)+1:offset(i+1));
                XP = xx(:,offset(j)+1:offset(j+1));
                yQ = yy(offset(i)+1:offset(i+1))';
                yP = yy(offset(j)+1:offset(j+1))';
                clear xx yy
                
            case 'amazon_small'
                
                red = 1:10;
                try
                    load('amazon.mat');
                catch
                    load('~/Data/Amazon/amazon.mat');
                end
                yy(yy==-1) = 2;
                
                XQ = xx(red,offset(i)+1:offset(i+1));
                XP = xx(red,offset(j)+1:offset(j+1));
                yQ = yy(offset(i)+1:offset(i+1))';
                yP = yy(offset(j)+1:offset(j+1))';
                clear xx yy
                
            case 'amazon_large'
                
                try
                    load('amazonR2.mat');
                catch
                    load('~/Data/Amazon/amazonR2.mat');
                end
                yy(yy==-1) = 2;
                
                XQ = X(dom==i,:)';
                XP = X(dom==j,:)';
                yQ = yy(dom==i)';
                yP = yy(dom==j)';
                clear X yy
                
            case 'amazon_msda'
                
                source = domains{i};
                fname = ['/tudelft.net/staff-bulk/ewi/insy/VisionLab/wkouw/mSDA/examples/amazon_msda_' source '.mat'];
                try
                    load(fname);
                catch
                    load(['/home/wmkouw/Dropbox/Projects/far/mSDA/examples/amazon_msda_' source '.mat'])
                end
                load('/tudelft.net/staff-bulk/ewi/insy/VisionLab/wkouw/amazon.mat', 'yy', 'offset');
                yy(yy==-1) = 2;
                
                XQ = allhx(:,offset(i)+1:offset(i+1));
                XP = allhx(:,offset(j)+1:offset(j+1));
                yQ = yy(offset(i)+1:offset(i+1))';
                yP = yy(offset(j)+1:offset(j+1))';
                clear allhx yy
                
        end
        
        % Shapes
        [MP,NP] = size(XP);
        [MQ,NQ] = size(XQ);
        
        switch prep
            case 'tf-idf'
                % Term frequency - inverse document frequency
                dfQ = log(NQ ./ (sum(XQ > 0, 2) + 1));
                XQ = bsxfun(@times, bsxfun(@rdivide, XQ, max(XQ, [], 1) + realmin), dfQ);
                dfP = log(NP ./ (sum(XP > 0, 2) + 1));
                XP  = bsxfun(@times, bsxfun(@rdivide,XP, max(XP, [], 1) + realmin), dfP);
            case 'fmax'
                % Normalize by maximum word use
                XQ = bsxfun(@rdivide, XQ, max(XQ, [], 2));
                XP = bsxfun(@rdivide, XP, max(XP, [], 2));
            case 'fsum'
                % Normalize by total word use
                XQ = bsxfun(@rdivide, XQ, sum(XQ,2) + realmin);
                XP = bsxfun(@rdivide, XP, sum(XP,2) + realmin);
        end
        
        % Set optimization parameters
        options.Display = 'valid';
        options.DerivativeCheck = 'off';
        options.method = 'lbfgs';
        options.useMex = 1;
        
        % Estimate corruption parameters
        switch lower(learner)
            case {'farj','farq','fdrj','fdrq'}
                if i == j
                    q = zeros(MQ+1,1);
                else
                    % constrained ML solution to dropout/blankout corruption
                    rP = mean(XP>0,2);
                    rQ = mean(XQ>0,2);
                    q = [max(1-rP./max(rQ,realmin),0);0];
                end
            case 'iwlr'
                if i == j
                    q = ones(1,NQ);
                else
                    % Instance reweighting coefficients
                    q = iw_lr(XQ, XP);
                end
            case 'iwlr2'
                if i == j
                    q = ones(1,NQ);
                else
                    % Instance reweighting coefficients
                    q = iw_lr2(XQ, XP);
                end
            otherwise
                q = [];
        end
        
        % Run with l2-regularization
        for l2 = 1:L2;
            disp(['Running regularizer ' num2str(lambda(l2))]);
            
            % Minimize loss
            switch lower(learner)
                case 'farj'
                    % Bugs
                    yQ(yQ==2) = 0;
                    W = minFunc(@FARj_grad, randn((MQ+1),1), options, [XQ; ones(1,NQ)], yQ, q, lambda(l2));
                    
                case 'farq'
                    yQ(yQ==2) = -1;
                    yP(yP==2) = -1;
                    W = minFunc(@FARq_grad, randn((MQ+1),1), options, [XQ; ones(1,NQ)], yQ, q, lambda(l2));
                    
                case 'fdrj'
                    yQ(yQ==2) = -1;
                    yP(yP==2) = -1;
                    W = minFunc(@FDRj_grad, randn((MQ+1),1), options, [XQ; ones(1,NQ)], yQ, q, lambda(l2));
                    
                case 'fdrq'
                    yQ(yQ==2) = -1;
                    yP(yP==2) = -1;
                    W = minFunc(@FDRq_grad, randn((MQ+1),1), options, [XQ; ones(1,NQ)], yQ, q, lambda(l2));
                    
                case 'mcf'
                    [lablist, ~, labels] = unique(yQ);
                    K = numel(lablist);
                    YQ = zeros(K, NQ);
                    YQ(sub2ind([K NQ], labels', (1:NQ))) = 1;
                    W = minFunc(@mcf_blankout_log_grad, randn((MQ+1)*nC,1), options, [XQ; ones(1,NQ)], YQ, q, lambda(l2));
                    
                case {'iwlr','iwlr2'}
                    W = minFunc(@mWLR_grad, randn((MQ+1)*nC,1), options, XQ, yQ, q, lambda(l2));
                    
                case 'logr'
                    W = minFunc(@mLR_grad, randn((MQ+1)*nC,1), options, XQ, yQ, lambda(l2));
                    
                case 'svm'
                    W = svmtrain(yQ',XQ',['-t 0 -c ',num2str(lambda(l2))]);
                    
            end
            
            % Evaluate learner
            switch lower(learner)
                case 'svm'
                    % Error on training set
                    [~,acc] = svmpredict(yQ',XQ',W);
                    trn_err(i,j,l2) = 1 - acc(1);
                    
                    % Error on test set
                    [~,acc] = svmpredict(yP',XP',W);
                    tst_err(i,j,l2) = 1 - acc(1);
                    
                case {'farq', 'farj', 'fdrj', 'fdrq'}
                    % Error on training set
                    trn_err(i,j,l2) = sum(sign(W'*[XQ; ones(1,NQ)]) ~= yQ)./length(yQ);
                    
                    % Error on test set
                    tst_err(i,j,l2) = sum(sign(W'*[XP; ones(1,NP)]) ~= yP)./length(yP);
                    
                otherwise
                    % Reshape weights
                    W0 = reshape(W(MQ*nC+1:end), [1 nC]);
                    W = [reshape(W(1:MQ*nC), [MQ nC]); W0];
                    
                    % Error on training set
                    [~,pred] = max(W'*[XQ; ones(1,size(XQ,2))], [], 1);
                    trn_err(i,j,l2) = sum(pred ~= yQ) ./ length(yQ);
                    
                    % Error on test set
                    [~,pred] = max(W'*[XP; ones(1,size(XP,2))], [], 1);
                    tst_err(i,j,l2) = sum(pred ~= yP) ./ length(yP);
                    
            end
            % Write results
            fn = ['results_' dataset '_' learner '_' prep '_' domains{i} '_' domains{j} '_lambda' num2str(lambda(l2)) '_' sv '.mat'];
            save(fn, 'trn_err', 'tst_err', 'W', 'q', '-v7.3');
        end
    end
end

exit;
end
