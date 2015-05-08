function [tme,tse,W,Theta,terr] = daexp_spam(clf,prep,varargin)
% Function to run some da experiments on spam mail/sms datasets

% Parse hyperparameters
p = inputParser;
addOptional(p, 'l2', 1e-3);
addOptional(p, 'nR', 1);
addOptional(p, 'nF', 1);
addOptional(p, 'fc', []);
addOptional(p, 'par', 1);
addOptional(p, 'save', 1);
parse(p, varargin{:});

% Really don't like long variable names...
l2 = p.Results.l2;
nR = p.Results.nR;
nF = p.Results.nF;
fc = p.Results.fc;

% Load data
try
    load('/home/nfs/wkouw/sm_spam.mat')
catch
    load('/home/wmkouw/Data/UCI-sms_spam/sm_spam.mat');
    %     load('/home/wmkouw/Data/UCI-sms_spam/sm_spam2.mat');
end

% In case of feature curve experiment, sample given features
if ~isempty(fc); sX = sX(fc,:); end

% Map labels to [1,2]
y(y==-1) = 2;

% Preprocess counts
sX = da_prep(sX,prep);

% Loop trough pairwise da combinations
lD = length(domain_names);
cmb = [nchoosek(1:lD,2); fliplr(nchoosek(1:lD,2)); repmat([1:lD], [2 1])'];
lCmb = length(cmb);

% Preallocation
W = cell(1,lCmb);
Theta = cell(1,lCmb);
tme = zeros(1,lCmb);
tse = zeros(1,lCmb);
terr = cell(1,lCmb);
for cc = 1:lCmb;
    
    % Update progress
    disp([domain_names{cmb(cc,1)} ' -> ' domain_names{cmb(cc,2)}]);
    
    % Slice source and target
    ixQ = domains(cmb(cc,1))+1:domains(cmb(cc,1)+1);
    ixP = domains(cmb(cc,2))+1:domains(cmb(cc,2)+1);
    XQ = [sX(:,ixQ); ones(1,length(ixQ))];
    XP = [sX(:,ixP); ones(1,length(ixP))];
    yQ = y(ixQ);
    yP = y(ixP);
    
    if cmb(cc,1) == cmb(cc,2);
        % Run a crossvalidation procedure on within-domain combination
        if p.Results.par;
            [W{cc},Theta{cc},terr{cc}] = da_crossval_within_par(clf,XQ,yQ,'nR',nR,'nF',nF,'l2',l2);
        else
            [W{cc},Theta{cc},terr{cc}] = da_crossval_within(clf,XQ,yQ,'nR',nR,'nF',nF,'l2',l2);
        end
    else
        % Run a crossvalidation procedure on between-domain combination
        if p.Results.par;
            [W{cc},Theta{cc},terr{cc}] = da_crossval_between_par(clf,XQ,yQ,XP,yP,'nR',nR,'nF',nF,'l2',l2);
        else
            [W{cc},Theta{cc},terr{cc}] = da_crossval_between(clf,XQ,yQ,XP,yP,'nR',nR,'nF',nF,'l2',l2);
        end
    end
    
    % Store the rep-fold errors
    tme(cc) = mean(mean(terr{cc},2),1);
    tse(cc) = std(mean(terr{cc},2))./sqrt(length(mean(terr{cc},2)));
    
end

if p.Results.save;
    % Write intermediate results
    if ~iscell(prep); prep = {prep}; end
    fname = ['daexp_spam_xval_'  clf '_' prep{:} '_' ...
        domain_names{cmb(cc,1)} '_' domain_names{cmb(cc,2)} '.mat'];
    save(fname, 'tme','tse','terr','Theta','W', 'cmb', 'l2');
end

end
