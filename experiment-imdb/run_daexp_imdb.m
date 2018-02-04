function run_daexp_imdb(clf)
% Script with experiment parameters

% Which pairwise domain combinations
cc = 1:6;

% Preprocessing 
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
daexp_imdb(clf, 'cix', cc, prep, 'nR', nR, 'nF', nF, 'l2', L2, 'nE', nE, 'mu', mu, ...
    'Kt', Kt, 'Kp', Kp, 'Ga', Ga, 'La', La);

end
