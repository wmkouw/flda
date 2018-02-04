function daexp_imdb(clf,varargin)
% Function to run experiments on the imdb dataset

% Parse hyperparameters
p = inputParser;
addOptional(p, 'cix', 1:6);
addOptional(p, 'prep', {});
addOptional(p, 'l2', 0);
addOptional(p, 'nR', 1);
addOptional(p, 'nE', 10);
addOptional(p, 'nF', 1);
addOptional(p, 'mu', 1);
addOptional(p, 'La', 1);
addOptional(p, 'Ga', 1);
addOptional(p, 'Kt', 'rbf');
addOptional(p, 'Kp', 1);
parse(p, varargin{:});

% Update progress
disp(['Evaluating ' clf ' on imdb']);

% Load data
load('da_imdb.mat');

% Preprocess counts
D = da_prep(D, p.Results.prep);

% Loop trough pairwise da combinations
lD = length(domain_names);
cmb = [nchoosek(1:lD,2); fliplr(nchoosek(1:lD,2))];
lCmb = length(cmb);

if any(strcmp(clf, {'tlr','tqd'}))
    
    % Preallocation
    W = cell(1,lD);
    Theta = cell(1,lD);
    err = zeros(1,lD);
    mis = cell(1,lD);
    lambda = zeros(1,lD);
    
    for d = 1:lD
        
        % Update progress
        disp([domain_names{d} ' -> ' domain_names{d}]);
        
        ixX = domains(d)+1:domains(d+1);
        X = [D(:,ixX); ones(1,length(ixX))];
        yX = y(ixX);
        [W{d},Theta{d},err(d),mis{d},lambda(d)] = da_xval(clf, X,yX,X,yX, ...
            'nR', p.Results.nR, 'nF', p.Results.nF, 'l2', p.Results.l2, ...
            'nE', p.Results.nE, 'mu', p.Results.mu, 'Kt', p.Results.Kt, ...
            'Kp', p.Results.Kp, 'La', p.Results.La, 'Ga', p.Results.Ga);
    end
    
else
    
    % Preallocation
    W = cell(1,lCmb);
    Theta = cell(1,lCmb);
    err = zeros(1,lCmb);
    mis = cell(1,lCmb);
    lambda = zeros(1,lCmb);
   
    % Loop through chosen combinations
    for cc = p.Results.cix
        
        % Update progress
        disp([domain_names{cmb(cc,1)} ' -> ' domain_names{cmb(cc,2)}]);
        
        % Slice source and target
        ixZ = domains(cmb(cc,2))+1:domains(cmb(cc,2)+1);
        ixX = domains(cmb(cc,1))+1:domains(cmb(cc,1)+1);
        Z = [D(:,ixZ); ones(1,length(ixZ))];
        X = [D(:,ixX); ones(1,length(ixX))];
        yZ = y(ixZ);
        yX = y(ixX);
        
        % Run a crossvalidation procedure for the l2 parameter
        [W{cc},Theta{cc}, err(cc), mis{cc}, lambda(cc)] = da_xval(clf,X,yX,Z,yZ, ...
            'nR', p.Results.nR, 'nF', p.Results.nF, 'l2', p.Results.l2, ...
            'nE', p.Results.nE, 'mu', p.Results.mu, 'Kt', p.Results.Kt, ...
            'Kp', p.Results.Kp, 'La', p.Results.La, 'Ga', p.Results.Ga);
        
    end
end

% Write results
fname = ['daexp_imdb_xval_'  clf '_' p.Results.prep{:} '.mat'];
disp(['Done. Writing to : ' fname]);
save(fname, 'err','Theta','W', 'cmb', 'mis', 'lambda', 'l2','p');

end

