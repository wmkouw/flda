function [serr,terr,W,theta] = daexp_digits_1vall(loss,transfer,prep)
% Function to run some da experiments on the MNIST vs SEMEION vs USPS datasets

% Add packages to path
addpath(genpath('minFunc'));

% To do:
% - transfer crossvalidation
disp(['Crossvalidation has not been implemented yet']);

% Options on loss and transfer are limited
if ~strcmp(transfer, {'dropout', 'blankout', ''});
    error(['Transfer distribution ' transfer ' is not supported']);
end

% Load data
try
    load('/home/nfs/wkouw/da_digits.mat')
catch
    load('/home/wmkouw/Data/MNIST/da_digits.mat');
end
y = y+1; % Go from 0-9, to 1-10
[N,M] = size(X);
K = numel(unique(y));

% Preprocess counts
X = da_prep(X,prep);

% Optimization options
options.DerivativeCheck = 'off';
options.Method = 'lbfgs';
options.Display = 'valid';

% Loop trough pairwise da combinations
lD = length(domain_names);
cmb = [nchoosek(1:lD,2); fliplr(nchoosek(1:lD,2)); repmat([1:lD], [2 1])'];
lCmb = length(cmb);

% Preallocation
W = cell(1,lCmb);
Theta = cell(1,lCmb);
serr = zeros(1,lCmb);
terr = zeros(1,lCmb);
l2 = 1e-3;
h1 = 30;
h2 = 25;
nE = 100;
for cc = 1:lCmb;
    
    % Update progress
    disp([domain_names{cmb(cc,1)} ' -> ' domain_names{cmb(cc,2)}]);
    
    % Slice source and target
    XQ = X(domains(cmb(cc,1))+1:domains(cmb(cc,1)+1),:)';
    XP = X(domains(cmb(cc,2))+1:domains(cmb(cc,2)+1),:)';
    yQ = y(domains(cmb(cc,1))+1:domains(cmb(cc,1)+1));
    yP = y(domains(cmb(cc,2))+1:domains(cmb(cc,2)+1));
    
    % Shapes
    [MQ,NQ] = size(XQ);
    [MP,NP] = size(XP);
    
    % Add bias
    XQ = [XQ; ones(1,NQ)];
    XP = [XP; ones(1,NP)];
    
    % Estimate transfer parameters
    switch transfer
        case 'dropout'
            theta = max(0,1-mean(XP>0,2)./mean(XQ>0,2));
        case 'blankout'
            theta = max(0,1-mean(XP>0,2)./mean(XQ>0,2));
        case ''
            theta = cell(1,K);
    end
    
    % Do one vs all classification (i.e. max confidence of K classifiers)
    if strcmp(loss, 'scl'); w = zeros(MQ+h2+1,K);
    else w = zeros(MQ+1,K); end
    
    for k = 1:K
        
        % Binarize labels for class k
        yk = reshape(double(yQ==k), [NQ 1]);
        yk(yk==0) = -1;
        
        % Upsample smaller class
        Xk = [XQ repmat(XQ(:,yQ==k), [1 K-1])];
        yk = [yk; ones((K-1)*sum(yk==1),1)];
        
        switch loss
            case 'square'
                switch transfer
                    case 'dropout'
                        w(:,k) = far_square_dropout(Xk,yk,theta,l2);
                    case 'blankout'
                        w(:,k) = far_square_blankout(Xk,yk,theta,l2);
                    otherwise
                        w(:,k) = (Xk*Xk'+l2*eye(M+1))\Xk*yk;
                end
                
            case 'logistic'
                switch transfer
                    case 'dropout'
                        w(:,k) = far_log_dropout(Xk,yk',theta,l2);
                    case 'blankout'
                        w(:,k) = far_log_blankout(Xk,yk',theta,l2);
                    otherwise
                        options.Display = 'valid';
                        w = minFunc(@mLR_grad, zeros((MQ+1)*K,1), options, Xk,yk',l2);
                        break;
                end
                
            case 'scl'
                [w,theta,~] = scl(XQ(1:end-1,:),XP(1:end-1,:),yQ,l2,h1,h2);
                XQ = [XQ(1:end-1,:); theta*XQ(1:end-1,:); ones(1,NQ)];
                XP = [XP(1:end-1,:); theta*XP(1:end-1,:); ones(1,NP)];
                break;
            case 'kmm'
                theta = irw_est(XQ,XP);
                w = irw_log(XQ,yQ,theta,l2);
                break;
            case 'irw'
                theta = irw_est(XQ,XP);
                w = irw_log(XQ,yQ,theta,l2);
                break;
            case 'sa'
                [w,EQ,EP] = subspace_alignment(XQ,XP,yQ,l2,nE);
                theta = {EQ,EP};
                XQ = [(XQ'*EQ*(EQ'*EP))'; ones(1,NQ)];
                XP = [(XP'*EP)'; ones(1,NP)];
                break;
            otherwise
                error(['Loss function ' loss ' not implemented']);
        end
    end
    
    % Create class-confidences
    confQ = w'*XQ;
    confP = w'*XP;
    
    % Evaluate classifier
    [~,predQ] = max(confQ, [], 1);
    [~,predP] = max(confP, [], 1);
    serr(cc) = sum(predQ'~=yQ)./length(yQ);
    terr(cc) = sum(predP'~=yP)./length(yP);
    
    % Store classifier
    W{cc} = w;
    
    % Store transfer parameters
    Theta{cc} = theta;
    
    % Write intermediate results
    fname = ['daexp_digits_'  loss '_' transfer '_' prep{:} '_' ...
        domain_names{cmb(cc,1)} '_' domain_names{cmb(cc,2)} '.mat'];
    save(fname, 'serr','terr','Theta','W', 'cmb', 'l2');
    
end

end
