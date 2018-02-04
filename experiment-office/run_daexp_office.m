function run_daexp_office(clf)
% Script with experiment parameters

% Which pairwise domain combinations
cc = 1:6;

% Which features to use
fts = 'caltech';

% Data preprocessing steps
prep = {'max'};

% Number of repetitions
nR = 1;

% Number of folds
nF = 2;

% Subspace dimensionality
nE = 500;

% Kernel type
Kt = 'rbf';

% Kernel bandwidth parameter
Kp = [1 10 100];

% Mu parameter for TCA
mu = [0.01 0.1 1 10];

% Lambda parameter
La = [0 1 10];

% Gamma parameter
Ga = [.00001 0.001 1];

% l2-regularization parameter
l2 = [1 10 100 1000 10000 100000];

% Experiment function
daexp_office(clf, 'cix', cc, 'prep', prep, 'nR', nR, 'nF', nF, 'l2', l2, ...
    'fts', fts, 'nE', nE, 'mu', mu, 'La', La, 'Ga', Ga, 'Kt', Kt, 'Kp', Kp);

end
