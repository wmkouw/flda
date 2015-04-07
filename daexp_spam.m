function [serr,terr,W,Theta] = daexp_spam(loss,transfer,prep)
% Function to run some da experiments on spam mail/sms datasets

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
    load('/home/nfs/wkouw/sm_spam.mat')
catch
%     load('/home/wmkouw/Data/UCI-sms_spam/sm_spam.mat');
     load('/home/wmkouw/Data/UCI-sms_spam/sm_spam2.mat');
end
[M,N] = size(sX);
K = numel(unique(y));

% Preprocess counts
sX = da_prep(sX,prep);

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
    XQ = sX(:,domains(cmb(cc,1))+1:domains(cmb(cc,1)+1));
    XP = sX(:,domains(cmb(cc,2))+1:domains(cmb(cc,2)+1));
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
            theta = [];
    end
    
    switch loss
        case 'square'
            switch transfer
                case 'dropout'
                    % First 2 moments of transfer distribution
                    W{cc} = far_square_dropout(XQ,yQ,theta,l2);
                case 'blankout'
                    % First 2 moments of transfer distribution
                    W{cc} = far_square_blankout(XQ,yQ,theta,l2);
                otherwise
                    % No transfer distribution, but we'll do regularization
                    W{cc} = (XQ*XQ'+l2*eye(MQ+1))\XQ*yQ;
            end
            
        case 'logistic'
            switch transfer
                case 'dropout'
                    W{cc} = far_log_dropout(XQ,yQ',theta,l2);
                case 'blankout'
                    W{cc} = far_log_blankout(XQ,yQ',theta,l2);
                otherwise
                    options.Display = 'valid';
                    w = minFunc(@mLR_grad, zeros((MQ+1)*K,1), options, Xk,yk',l2);
                    W{cc} = w([1:MQ end-1]);
            end
        case 'scl'
            [W{cc},theta,~] = scl(XQ(1:end-1,:),XP(1:end-1,:),yQ,l2,h1,h2);
            XQ = [XQ(1:end-1,:); theta*XQ(1:end-1,:); ones(1,NQ)];
            XP = [XP(1:end-1,:); theta*XP(1:end-1,:); ones(1,NP)];
        case 'kmm'
            theta = irw_est(XQ,XP);
            w = irw_log(XQ,yQ,theta,l2);
            W{cc} = w([1:MQ end-1]);
        case 'irw'
            theta = irw_est(XQ,XP);
            w = irw_log(XQ,yQ,theta,l2);
            W{cc} = w([1:MQ end-1]);
        case 'sa'
            [W{cc},EQ,EP] = subspace_alignment(XQ,XP,yQ,l2,nE);
            theta = {EQ,EP};
            XQ = [(XQ'*EQ*(EQ'*EP))'; ones(1,NQ)];
            XP = [(XP'*EP)'; ones(1,NP)];
        otherwise
            error(['Loss function ' loss ' not implemented']);
    end
    
    % Evaluate classifier
    serr(cc) = sum(sign(W{cc}'*XQ)~=yQ')./length(yQ);
    terr(cc) = sum(sign(W{cc}'*XP)~=yP')./length(yP);
    
    % Store transfer parameters
    Theta{cc} = theta;
    
    % Write intermediate results
    fname = ['daexp_spam_'  loss '_' transfer '_' prep{:} '_' ...
        domain_names{cmb(cc,1)} '_' domain_names{cmb(cc,2)} '.mat'];
    save(fname, 'serr','terr','Theta','W', 'cmb', 'l2');
    
end

end
