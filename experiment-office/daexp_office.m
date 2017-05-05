function daexp_office(clf,varargin)
% Function to run experiments on the office dataset with surf features

% Parse hyperparameters
p = inputParser;
addOptional(p, 'cix', []);
addOptional(p, 'prep', {});
addOptional(p, 'fts', 'surf');
addOptional(p, 'l2', 0);
addOptional(p, 'nR', 1);
addOptional(p, 'nF', 1);
addOptional(p, 'nE', 100);
addOptional(p, 'mu', 1);
addOptional(p, 'La', 1);
addOptional(p, 'Ga', 1);
addOptional(p, 'Kt', 'rbf');
addOptional(p, 'Kp', 1);
parse(p, varargin{:});

% Really don't like long variable names...
l2 = p.Results.l2;
nR = p.Results.nR;
nF = p.Results.nF;
nE = p.Results.nE;
mu = p.Results.mu;
La = p.Results.La;
Ga = p.Results.Ga;
Kt = p.Results.Kt;
Kp = p.Results.Kp;
prep = p.Results.prep;

% Update progress
disp(['Evaluating ' clf ' on office-' p.Results.fts]);

% Which features to use
switch p.Results.fts
    case 'caltech'
        % Load data
        try
            load('/tudelft.net/staff-bulk/ewi/insy/PRLab/Staff/wmkouw/flda-office/office_caltech01.mat')
        catch
            load('office_caltech01.mat');
        end
        disp(['Using Office-Caltech SURF features']);
    case 'surf'
        % Load data
        try
            load('/tudelft.net/staff-bulk/ewi/insy/PRLab/Staff/wmkouw/flda-office/office_surf.mat')
        catch
            load('office_surf.mat');
        end
        disp(['Using SURF features']);
        
    case 'cnn_fc6'
        % Load data
        try
            load('/tudelft.net/staff-bulk/ewi/insy/PRLab/Staff/wmkouw/flda-office/office_decafe_fc6.mat')
        catch
            load('office_decafe_fc6.mat');
        end
        disp(['Using layer 6 CNN features']);
        
    case 'cnn_fc7'
        % Load data
        try
            load('/tudelft.net/staff-bulk/ewi/insy/PRLab/Staff/wmkouw/flda-office/office_decafe_fc7.mat')
        catch
            load('office_decafe_fc7.mat');
        end
        disp(['Using layer 7 CNN features']);
        
    case 'cnn_fc8'
        % Load data
        try
            load('/tudelft.net/staff-bulk/ewi/insy/PRLab/Staff/wmkouw/flda-office/office_decafe_fc8.mat')
        catch
            load('office_decafe_fc8.mat');
        end
        disp(['Using layer 8 CNN features']);
end

% Preprocess counts
D = da_prep(D',prep);

% Loop trough pairwise da combinations
lD = length(domain_names);
cmb = [nchoosek(1:lD,2); fliplr(nchoosek(1:lD,2))];
lCmb = length(cmb);

% Check for index of source-target combination
if ~isempty(p.Results.cix)
    lcc = p.Results.cix;
    lCmb = 1;
else
    lcc = 1:lCmb;
end

if any(strcmp(clf, {'tlr','tqd'}));
    if nF==1; error('Crossvalidation necessary for target classifier'); end
    
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
        [W{d},Theta{d},err(d),mis{d},lambda(d)] = da_xval(clf, X,yX,X,yX,'nR', nR, 'nF', nF, 'l2', l2, 'nE', nE,'mu',mu, 'La', La, 'Ga', Ga,'Kt',Kt,'Kp',Kp);
    end
    
else
    
    % Preallocation
    W = cell(1,lCmb);
    Theta = cell(1,lCmb);
    err = zeros(1,lCmb);
    mis = cell(1,lCmb);
    lambda = zeros(1,lCmb);
    
    
    % Loop through all pairwise combinations of domains
    for cc = lcc;
        
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
        [W{cc},Theta{cc}, err(cc), mis{cc}, lambda(cc)] = da_xval(clf,X,yX,Z,yZ,'nR', nR, 'nF', nF, 'nE', nE, 'l2', l2,'mu',mu, 'La', La, 'Ga', Ga, 'Kt',Kt,'Kp',Kp);
        
    end
end

% Write results
fname = ['daexp_office_' p.Results.fts '_xval_'  clf '_prep' prep{:} '_cix' num2str(p.Results.cix) '.mat'];
disp(['Done. Writing to : ' fname]);
save(fname, 'err','Theta','W', 'cmb', 'mis','lambda', 'l2','p');

end

