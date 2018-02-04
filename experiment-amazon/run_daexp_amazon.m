function run_daexp_amazon(clf)
% Script with experiment parameters

% Which pairwise domain combinations
cc = 1:12;

% Preprocessing steps
prep = {'max'};

% Number of repetitions
nR = 1;

% Number of folds
nF = 2;

% Number of subspace dimensions
nE = 500;

% Type of kernel
Kt = 'rbf';

% Kernel bandwidth parameter 
Kp = [.001 1 1000];

% Mu parameter for TCA
mu = [0.01 0.1 1 10];

% Lambda parameter
La = [0 1 10];

% Gamma parameter
Ga = [.00001 0.001 1];

% l2-regularization parameter
l2 = [1 10 100 1000 10000 100000];

% Experiment function
daexp_amazon(clf, 'prep', prep, 'cix', cc, 'nR', nR, 'nF', nF, 'l2', l2, ...
    'nE', nE, 'mu', mu,'Kt',Kt,'Kp',Kp,'Ga',Ga,'La',La);

end
